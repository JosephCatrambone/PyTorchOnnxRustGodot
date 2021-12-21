# PyTorchOnnxRustGodot
A prototype to add a trained PyTorch model to Godot via Onnx and Rust+GDNative.

# Directory Structure and files:
The root folder (`/`) is a Rust Project (made via `cargo new pix2cat --lib`).  Inside of it a new Godot Project is created.

`/models/pix2cat.onnx` would normally reside in the models folder and be packaged into the GDNative library at Rust build time.  It has been replaced by two files, copied straight from https://github.com/JosephCatrambone/unet just so the whole project is self contained.

`src/lib.rs` and `Cargo.toml` are the required pieces to build the interface and wrap the ONNX model in a form that's usable.  When `cargo build --release` is invoked, a DLL is created in the targets folder which can be linked into Godot by the GDNative script.  The lib.rs uses GDNative to specify a Node subclass called `Pix2Cat`, which can be invoked through GDScript.

`MainGame.tscn` sets up the UI for the application, two TextureImages and a few buttons.  `MainGame.gd` has all of the logic, which is 90% simply supporting drawing to the frame and copy data, with two lines actually dedicated to calling the ONNX model.

As a nice advantage to this setup, the lib.rs file uses include_bytes!, so there's no need to manage a standalone resource alongside the game.  This should work if one is building for mobile platforms, too, but this is untested.
