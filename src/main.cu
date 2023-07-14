/*
 -*- coding: utf-8 -*-
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Contact: insta@tue.mpg.de
*/

#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <rta/core.h>

#include <args/args.hxx>

#include <filesystem/path.h>

// billboards
#include <thread>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <sstream>
#include <vector>
#include <functional>

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
namespace fs = ::filesystem;


// billboards
typedef websocketpp::server<websocketpp::config::asio> server;
rta::Core core(ETestbedMode::Nerf);
server s;

Eigen::Matrix<float, 3, 4> convertToMatrix(const std::string& input) {
    std::stringstream ss(input);
    float value;
    std::vector<float> values;
    while (ss >> value) {
        values.push_back(value);
        if (ss.peek() == ',')
            ss.ignore();
    }
    Eigen::Matrix<float, 3, 4> result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            result(i, j) = values[j * 4 + i];
        }
    }
    return result;
}

void on_message(websocketpp::connection_hdl hdl, server::message_ptr msg) {
    Eigen::Matrix<float, 3, 4> mat = convertToMatrix(msg->get_payload());

    // ref: rta::Recorder::video()
    core.m_recorder->m_ngp->m_target_deform_frame = core.m_recorder->m_index_frame;
    core.m_recorder->m_ngp->m_nerf.extra_dim_idx_for_inference = core.m_recorder->m_index_frame;
    core.m_recorder->m_dst_folder = "floating";
    core.m_recorder->m_to_record = core.m_nerf.training.dataset.n_all_images;

    // ref: rta::Recorder::start()
    core.m_dynamic_res = false;

    core.m_camera = mat;
    core.m_zoom = 5.f;
    std::vector<uint8_t> pngpixels = core.m_recorder->dump_frame_buffer();
    try {
        s.send(hdl, pngpixels.data(), pngpixels.size(), websocketpp::frame::opcode::binary);
    } catch (websocketpp::exception const & e) {
        std::cout << "Error sending message: " << e.what() << std::endl;
    }
}

void start_server(server& s) {
    s.clear_access_channels(websocketpp::log::alevel::all);
    s.clear_error_channels(websocketpp::log::elevel::all);
    s.set_message_handler(&on_message);
    s.init_asio();
    s.listen(9002);
    s.start_accept();
    s.run();
}

int main(int argc, char **argv) {
    ArgumentParser parser{
            "neural graphics primitives\n"
            "version " NGP_VERSION,
            "",
    };

    HelpFlag help_flag{
            parser,
            "HELP",
            "Display this help menu.",
            {'h', "help"},
    };

    ValueFlag<string> mode_flag{
            parser,
            "MODE",
            "Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.",
            {'m', "mode"},
    };

    ValueFlag<string> network_config_flag{
            parser,
            "CONFIG",
            "Path to the network config. Uses the scene's default if unspecified.",
            {'n', 'c', "network", "config"},
    };

    Flag no_gui_flag{
            parser,
            "NO_GUI",
            "Disables the GUI and instead reports training progress on the command line.",
            {"no-gui"},
    };

    Flag no_train_flag{
            parser,
            "NO_TRAIN",
            "Disables training on startup.",
            {"no-train"},
    };

    ValueFlag<string> scene_flag{
            parser,
            "SCENE",
            "The scene to load. Can be NeRF dataset, a *.obj mesh for training a SDF, an image, or a *.nvdb volume.",
            {'s', "scene"},
    };

    ValueFlag<string> snapshot_flag{
            parser,
            "SNAPSHOT",
            "Optional snapshot to load upon startup.",
            {"snapshot"},
    };

    ValueFlag<uint32_t> width_flag{
            parser,
            "WIDTH",
            "Resolution width of the GUI.",
            {"width"},
    };

    ValueFlag<uint32_t> height_flag{
            parser,
            "HEIGHT",
            "Resolution height of the GUI.",
            {"height"},
    };

    Flag version_flag{
            parser,
            "VERSION",
            "Display the version of neural graphics primitives.",
            {'v', "version"},
    };

    // Parse command line arguments and react to parsing
    // errors using exceptions.
    try {
        parser.ParseCLI(argc, argv);
    } catch (const Help &) {
        cout << parser;
        return 0;
    } catch (const ParseError &e) {
        cerr << e.what() << endl;
        cerr << parser;
        return -1;
    } catch (const ValidationError &e) {
        cerr << e.what() << endl;
        cerr << parser;
        return -2;
    }

    if (version_flag) {
        tlog::none() << "neural graphics primitives version " NGP_VERSION;
        return 0;
    }

    try {
        ETestbedMode mode = ETestbedMode::Nerf;
        if (!mode_flag) {
            if (!scene_flag) {
                tlog::error() << "Must specify either a mode or a scene";
                return 1;
            }

            fs::path scene_path = get(scene_flag);
            if (!scene_path.exists()) {
                tlog::error() << "Scene path " << scene_path << " does not exist.";
                return 1;
            }

            if (scene_path.is_directory() || equals_case_insensitive(scene_path.extension(), "json")) {
                mode = ETestbedMode::Nerf;
            } else if (equals_case_insensitive(scene_path.extension(), "obj") || equals_case_insensitive(scene_path.extension(), "stl")) {
                mode = ETestbedMode::Sdf;
            } else if (equals_case_insensitive(scene_path.extension(), "nvdb")) {
                mode = ETestbedMode::Volume;
            } else {
                mode = ETestbedMode::Image;
            }
        } else {
            auto mode_str = get(mode_flag);
            if (equals_case_insensitive(mode_str, "nerf")) {
                mode = ETestbedMode::Nerf;
            } else if (equals_case_insensitive(mode_str, "sdf")) {
                mode = ETestbedMode::Sdf;
            } else if (equals_case_insensitive(mode_str, "image")) {
                mode = ETestbedMode::Image;
            } else if (equals_case_insensitive(mode_str, "volume")) {
                mode = ETestbedMode::Volume;
            } else {
                tlog::error() << "Mode must be one of 'nerf', 'sdf', 'image', and 'volume'.";
                return 1;
            }
        }

        // billboards
        // rta::Core core(mode);

        std::string mode_str;
        switch (mode) {
            case ETestbedMode::Nerf:
                mode_str = "nerf";
                break;
            case ETestbedMode::Sdf:
                mode_str = "sdf";
                break;
            case ETestbedMode::Image:
                mode_str = "image";
                break;
            case ETestbedMode::Volume:
                mode_str = "volume";
                break;
        }

        // Otherwise, load the network config and prepare for training
        fs::path network_config_path = fs::path{"configs"} / mode_str;
        if (network_config_flag) {
            auto network_config_str = get(network_config_flag);
            if ((network_config_path / network_config_str).exists()) {
                network_config_path = network_config_path / network_config_str;
            } else {
                network_config_path = network_config_str;
            }
        } else {
            network_config_path = network_config_path / "base.json";
        }

        if (!network_config_path.exists()) {
            tlog::error() << "Network config path " << network_config_path << " does not exist.";
            return 1;
        }

        core.reload_network_from_file(network_config_path.str());

        if (scene_flag) {
            fs::path scene_path = get(scene_flag);
            if (!scene_path.exists()) {
                tlog::error() << "Scene path " << scene_path << " does not exist.";
                return 1;
            }
            core.load_training_data(scene_path.str());
        }

        core.m_train = !no_train_flag;
        core.post_loading();

        if (snapshot_flag) {
            // Load network from a snapshot if one is provided
            fs::path snapshot_path = get(snapshot_flag);
            if (!snapshot_path.exists()) {
                tlog::error() << "Snapshot path " << snapshot_path << " does not exist.";
                return 1;
            }

            core.load_snapshot(snapshot_path.str());

            core.m_train = false;
            core.m_offscreen_rendering = false;
        }

        bool gui = !no_gui_flag;
#ifndef NGP_GUI
        gui = false;
#endif
        // auto W = width_flag ? get(width_flag) : 1024;
        // auto H = height_flag ? get(height_flag) : 1024;
        auto W = 800;
        auto H = 800;

        if (gui) {
            core.init_window(W, H);
        } else {
            core.init_render_surface(W, H);
        }

        core.is_using_gui = gui;

        // billboards
        std::thread server_thread(start_server, std::ref(s));

        // Render/training loop
        while (core.frame()) {
            if (!gui) {
                if (core.is_recording())
                    tlog::info() << core.get_recorder_info();
                else
                    tlog::info() << "iteration=" << core.m_training_step << " loss=" << core.m_loss_scalar.val();
            }
        }
    } catch (const exception &e) {
        tlog::error() << "Uncaught exception: " << e.what();
        return 1;
    }
}
