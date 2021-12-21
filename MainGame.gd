extends Control

onready var pix2cat = preload("res://Pix2Cat.gdns").new()
onready var input:TextureRect = $UI/InputCenter/InputTextureRect
onready var output:TextureRect = $UI/OutputCenter/OutputTextureRect
onready var run_button:Button = $UI/VBoxContainer/RunButton
onready var clear_button:Button = $UI/VBoxContainer/ClearButton
onready var autorefresh:CheckBox = $UI/VBoxContainer/AutoRefresh

var canvas_dirty:bool = false
var drawing:bool = false
var input_image:Image
var brush_size:int = 2


func _ready():
	make_image()
	update_input_texture()
	run_button.connect("pressed", self, "run_net")
	clear_button.connect("pressed", self, "clear_image")

func _process(delta):
	if self.canvas_dirty:
		self.update_input_texture()
		self.canvas_dirty = false
		if not self.drawing and autorefresh.is_pressed():
			print("Autorefreshing")
			run_net()

func _input(event):
	if event is InputEventMouseButton:  # Pressed?
		self.drawing = event.pressed
	elif event is InputEventMouseMotion and self.drawing:
		var pixelx = input_image.get_width()*(float(event.position.x-input.rect_position.x)/float(input.rect_size.x))  # This is local, right?
		var pixely = input_image.get_height()*(float(event.position.y-input.rect_position.y)/float(input.rect_size.y))
		input_image.lock()
		for dy in range(-self.brush_size, self.brush_size):
			for dx in range(-self.brush_size, self.brush_size):
				if pixelx+dx < 0 or pixelx+dx >= input_image.get_width() or pixely+dy < 0 or pixely+dy >= input_image.get_height():
					continue
				input_image.set_pixel(pixelx+dx, pixely+dy, Color.black)
		input_image.unlock()
		self.canvas_dirty = true

func make_image():
	self.input_image = Image.new()
	self.input_image.create(128, 128, false, Image.FORMAT_RGB8)
	self.clear_image()

func clear_image():
	self.input_image.fill(Color.white)

func update_input_texture():
	var imgtex = ImageTexture.new()
	imgtex.create_from_image(input_image)
	self.input.texture = imgtex

func run_net():
	var output_image:Image = pix2cat.pix_to_cat(input_image)
	var output_texture = ImageTexture.new()
	output_texture.create_from_image(output_image)
	output.texture = output_texture
	print("Net run.")
