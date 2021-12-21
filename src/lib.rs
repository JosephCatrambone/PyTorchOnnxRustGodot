use gdnative::api as gdapi;
use gdnative::core_types::{ByteArray, FromVariant};
use gdnative::{GodotObject, Ref, TRef, godot_init};
use gdnative::methods;
use gdnative::NativeClass;
use gdnative::nativescript::InitHandle;
use gdnative::prelude::TypedArray;
use gdnative::thread_access::{Shared, Unique};
use gdnative_derive::*;
use tract_onnx;
use tract_onnx::model::Onnx;
use tract_onnx::prelude::{Datum, Framework, Graph, InferenceFact, InferenceModel, SimplePlan, Tensor, tract_ndarray, tvec, TypedFact, TypedOp, InferenceModelExt};
use std::io::Cursor;
use std::ops::Deref;

const MODEL_INPUT_CHANNELS: usize = 1;
const MODEL_INPUT_HEIGHT: usize = 128;
const MODEL_INPUT_WIDTH: usize = 128;
const MODEL_OUTPUT_CHANNELS: i64 = 3;
const MODEL_OUTPUT_HEIGHT: i64 = 128;
const MODEL_OUTPUT_WIDTH: i64 = 128;

#[derive(NativeClass)]
#[inherit(gdapi::node::Node)]
struct Pix2Cat {
	// It's possible to export any type that implements `Export`, `ToVariant` and `FromVariant` using `#[property]`
	// All these traits are implemented for `Instance<T, Shared>` where the base class of `T` is reference-counted.
	// `Resource` inherits from `Reference`, so all native scripts extending `Resource` have reference-counted base classes.
	// Example:
	//#[property]
	//greeting_resource: Option<Instance<GreetingResource, Shared>>,
	// However, we want this to be opaque to Godot.
	model: Option<SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>>>
}

#[methods]
impl Pix2Cat {
	fn new_rust() -> Self {
		let model_bytes = include_bytes!("../models/drawing_to_cat.onnx");
		let mut model_reader = Cursor::new(model_bytes);

		/*
		let mut inf_model = tract_onnx::onnx().model_for_read(&mut model_reader).expect("Can't load model from embedded.");
		for n in &inf_model.nodes {
			dbg!("{:?} -> {:?}", &n.id, &n.name);
		}
		let input_node_idx = inf_model.node_by_name("input").expect("Couldn't find input node.").id;
		let output_node_idx = inf_model.node_by_name("output").expect("Couldn't find output node.").id;
		dbg!("{:?}", inf_model.output_fact(0));
		 */

		Pix2Cat {
			model: Some(
				tract_onnx::onnx()
					// load the model
					.model_for_read(&mut model_reader)//.model_for_path("models/encoder_cpu.onnx")
					.expect("Failed to load model from models/drawing_to_animal.onnx")
					// specify input type and shape
					.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, MODEL_INPUT_CHANNELS as i64, MODEL_INPUT_HEIGHT as i64, MODEL_INPUT_WIDTH as i64)))
					.expect("Failed to specify input shape.")
					// Make sure output shape is defined:
					//.with_output_fact(output_node_idx, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, MODEL_OUTPUT_CHANNELS as i64, MODEL_OUTPUT_HEIGHT as i64, MODEL_OUTPUT_WIDTH as i64)))
					//.expect("Failed to specify output shape.")
					// Quantize
					.into_optimized()
					.expect("Unable to optimize model for inference")
					// make the model runnable and fix its inputs and outputs
					.into_runnable()
					.expect("Failed make model runnable.")
			)
		}
	}

	fn new(_owner: &gdapi::node::Node) -> Self {
		Pix2Cat::new_rust()
	}

	#[export]
	fn _ready(&self, _owner: &gdapi::node::Node) {
		// Load model.
		/*
		if let Some(greeting_resource) = self.greeting_resource.as_ref() {
			let greeting_resource = unsafe { greeting_resource.assume_safe() };
			greeting_resource.map(|s, o| s.say_hello(&*o)).unwrap();
		}
		*/
	}

	#[export]
	fn pix_to_cat(&self, _owner: &gdapi::node::Node, image: Ref<gdapi::Image>) -> Ref<gdapi::Image, Unique> {
		//fn pix_to_cat(&self, _owner: &gdapi::node::Node, image_data: Ref<ByteArray>, image_width: i64, image_height: i64) -> Ref<ByteArray> {

		let img_ref: TRef<gdapi::Image> = unsafe { image.assume_safe() };
		let img: &gdapi::Image = img_ref.deref();
		let data:TypedArray<u8> = img.get_data();
		let img_width = img.get_width();
		let img_height = img.get_height();

		// image is an rgb8 but our model expects u8
		//let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
		let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, MODEL_INPUT_CHANNELS, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), |(_, c, y, x)| {
			// Assume [r g b a r g b a r g b a ...] row major.
			//data[(x + y*MODEL_INPUT_WIDTH)*3 + c] as f32 / 255.0
			data.get(((x + y*MODEL_INPUT_WIDTH)*MODEL_INPUT_CHANNELS + c) as i32) as f32 / 255.0
		}).into();

		// run the model on the input
		let result = if let Some(mdl) = &self.model {
			Some(mdl.run(tvec!(image)).unwrap())
		} else {
			None
		};

		let output_image: Vec<u8> = result.unwrap()[0]
			.to_array_view::<f32>().unwrap()
			.iter()
			.map(|v|{ (v.max(0f32).min(1f32) * 255f32) as u8 })
			.collect();

		// Output image is now in CHW form.  We need to convert to WHC.
		let mut converted_output: Vec<u8> = Vec::<u8>::with_capacity((MODEL_OUTPUT_CHANNELS*MODEL_OUTPUT_WIDTH*MODEL_OUTPUT_HEIGHT) as usize);
		//for idx in 0..output_image.len() {}
		// value(n, c, h, w) = n * CHW + c * HW + h * W + w
		// offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
		// offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
		// Convert this value index from CHW, [c*(w*h) + y*width + x] to WHC/RGB [(x+y*w)*3 + c]
		for y in 0..MODEL_OUTPUT_HEIGHT {
			for x in 0..MODEL_OUTPUT_WIDTH {
				for c in 0..MODEL_OUTPUT_CHANNELS {
					// Get this position in the output_image and append it to our RGB image.
					let original_offset = (c*MODEL_OUTPUT_HEIGHT*MODEL_OUTPUT_WIDTH) + (y*MODEL_OUTPUT_WIDTH) + x;
					converted_output.push(output_image[original_offset as usize]);
				}
			}
		}
		
		let i = gdapi::Image::new();
		i.create_from_data(
			MODEL_OUTPUT_WIDTH,
			MODEL_OUTPUT_HEIGHT,
			false,
			gdapi::Image::FORMAT_RGB8,
			TypedArray::from_vec(converted_output)
		);
		i
	}
}

fn init(handle: InitHandle) {
	handle.add_class::<Pix2Cat>();
}

godot_init!(init);

#[cfg(test)]
mod tests {
	use gdnative::api as gdapi;
	use crate::Pix2Cat;

	#[test]
	fn it_works() {
		let result = 2 + 2;
		assert_eq!(result, 4);
	}

	#[test]
	fn instance_test() {
		let pix2cat = Pix2Cat::new_rust();
	}
}
