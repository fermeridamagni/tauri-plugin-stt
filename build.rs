const COMMANDS: &[&str] = &[
    "start_listening",
    "stop_listening",
    "is_available",
    "get_supported_languages",
    "check_permission",
    "request_permission",
    "register_listener",
    "remove_listener",
    "list_models",
    "install_model",
    "remove_model",
    "set_active_model",
    "unload_model",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .ios_path("ios")
        .build();
}
