import numpy
import os
import sys
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
import torchvision.transforms as transforms
from glob import glob
from tqdm import tqdm

# For recording progress and showing outputs:
from torch.utils.tensorboard import SummaryWriter

from datasets.sketch_to_picture_dataset import SketchToPictureDataset
from model import UNet

#wandb.init(project="drawing_to_art", entity="josephc")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "drawing_to_cat"
DATASET_NAME = "cats"
NUM_WORKERS = 4
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 16
CHANGENOTES = "Previous run failed because an image was smaller than the min crop.  Random crop, more transforms, bump learning rate and epochs back to 100."


def record_run_config(filename, output_dir) -> int:
	"""Return the number of previous runs."""
	previous_runs = len(glob(f"{output_dir}/{filename[:-3]}*"))
	run_number = previous_runs+1  # One indexed.  Whatever.
	with open(os.path.join(output_dir, f"{filename}_{run_number}"), 'wt') as fout:
		fout.write(f"MODEL NAME: {MODEL_NAME}\n")
		fout.write(f"DATASET NAME: {DATASET_NAME}\n")
		fout.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
		fout.write(f"EPOCHS: {EPOCHS}\n")
		fout.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
		fout.write(f"CHANGENOTES: {CHANGENOTES}")
	return run_number


def export_model(model, input_channels, input_height, input_width, filename):
	model.eval()
	x = torch.randn(1, input_channels, input_height, input_width, requires_grad=True)
	_output = model(x)
	torch.onnx.export(
		model,
		x,
		filename,
		export_params=True,
		opset_version=10,
		do_constant_folding=True,
		input_names=['input'],
		output_names=['output'],
		dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
	)
	model.train()


def train(dataset, model, optimizer, loss_fn, summary_writer=None):
	for epoch_idx in range(EPOCHS):
		dataloop = tqdm(dataset)
		total_epoch_loss = 0.0
		for batch_idx, (data, targets) in enumerate(dataloop):
			step = (epoch_idx * len(dataloop)) + batch_idx
			data = data.unsqueeze(1).to(device=DEVICE)  # NOTE: Input is greyscale, so we unsqueeze channels at 1.
			tgt = targets.float().permute(0, 3, 1, 2).to(device=DEVICE)
			optimizer.zero_grad()

			# Forward
			preds = model(data)

			# Backward
			loss = loss_fn(preds, tgt)
			loss.backward()
			optimizer.step()

			# Log status.
			total_epoch_loss += loss.item()
			#wandb.log({"loss": loss})
			#wandb.watch(model)
			if summary_writer and batch_idx % 100 == 0:
				# Save sample images.
				input_grid = torchvision.utils.make_grid(data)
				summary_writer.add_image("Input Grid", input_grid, step)
				target_grid = torchvision.utils.make_grid(tgt)
				summary_writer.add_image("Target Grid", target_grid, step)
				output_grid = torchvision.utils.make_grid(preds)
				summary_writer.add_image("Output Grid", output_grid, step)
				# matplotlib_imshow(input_grid)
				# matplotlib_imshow(result_grid)

				# Write all network params to the log.
				#for name, weight in model.named_parameters():
				#	summary_writer.add_histogram(name, weight, step)
				#	summary_writer.add_histogram(f'{name}.grad', weight.grad, step)

				summary_writer.add_scalars('Training Loss', {"Last Training Loss": loss.item(), "Running Loss": total_epoch_loss}, step)
				total_epoch_loss = 0.0
				summary_writer.flush()

		torch.save(model.state_dict(), f"models/{MODEL_NAME}_{epoch_idx}")
		#export_model(model, 1, 128, 128, f"models/{MODEL_NAME}_{epoch_idx}.onnx")


def main(model_start_file=None):
	model = UNet(in_channels=1, out_channels=3).to(device=DEVICE)
	loss_fn = nn.L1Loss()
	optimizer = opt.Adam(model.parameters(), lr=LEARNING_RATE)

	if model_start_file:
		print(f"Restarting from checkpoint {model_start_file}")
		model.load_state_dict(torch.load(model_start_file))

	transform = transforms.Compose([
		#transforms.Normalize((0.5,), (0.5,)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(20),
		transforms.Resize((256,256)),
		transforms.RandomCrop((128, 128)),
		#transforms.ToTensor(),  # Don't do a ToTensor conversion at the end.
	])

	dataset = SketchToPictureDataset(base_image_folder=f"datasets/{DATASET_NAME}", transform=transform, target_width=128, target_height=128)
	training_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

	# Set up summary writer and record run stats.
	run_number = record_run_config(MODEL_NAME, "runs")
	os.mkdir(os.path.join("runs", str(run_number)))
	summary_writer = SummaryWriter(f"runs/{run_number}")
	print(f"Writing summary log to runs/{run_number}")

	# Log the model architecture:
	summary_writer.add_graph(model, torch.Tensor(numpy.zeros((1,1,dataset.target_height,dataset.target_width))).to(DEVICE))

	# Train
	train(training_loader, model, optimizer, loss_fn, summary_writer)

	# Write to file.
	export_model(model, 1, dataset.target_height, dataset.target_width, "result_model.onnx")


if __name__=="__main__":
	starting_file = None
	if len(sys.argv) > 1:
		starting_file = sys.argv[1]
	main(starting_file)
