extends Node3D
# EMIL Editor: bot on a floor, orbit camera (drag, pinch), tools, settings panel, core timing.
# Everything is built in code to keep the project small. Drawing: 3 MultiMeshes fed by EmilWorld.get_buffer().

# No curl tool: the curl comes by itself (rolling, high falls, car).
const TOOLS := ["Толчок", "Пуля", "Подножка", "Высота", "Машина", "Шаги"]
const TOOL_STEPS := 5
const STEP_FWD := 0.25   # commanded step lengths [m] (step ladder)
const STEP_BACK := -0.25
const STEP_SIDE := 0.20
const STEP_CROSS_DX := 0.10
const STEP_CROSS_DY := -0.34  # across: stance width 0.24 m + 0.10 m past the other ankle
const SINGLE_LEG_S := 3.0
# Core parameters with a slider in the settings panel: name -> label (order = panel order).
const BALANCE_PARAMS := {
	"assist": "помощь устойчивости",
	"stiff_scale": "жёсткость тела",
	"leg_k": "жёсткость ног",
	"step_h": "высота шага",
	"step_len": "длина шага",
	"give_up": "порог сдачи",
	"knee_weak": "ослабление колена",
	"step_budget": "запас шагов",
	"upper": "верх тела вкл/выкл",
	"up_back": "жёсткость спины",
	"up_arm": "жёсткость рук",
	"up_swing": "мах рук",
	"up_sway": "отставание рук",
	"up_twist": "скрутка спины",
	"up_spread": "руки в стороны",
	"up_loose": "расхлябанность",
	"up_loose_t": "срок расхлябанности",
	"up_gaze": "взгляд",
	"bul_lean": "наклон от пули",
	"bul_crouch": "присед от пули",
	"bul_imp": "импульс пули, Н·с",
}
const BURST_N := 4        # bullets in a burst
const BURST_DT := 0.15    # s between the bullets of a burst (sim time)

# Material 3 baseline palette (light).
const C_PRIMARY := Color("#6750A4")
const C_ON_PRIMARY := Color("#FFFFFF")
const C_PRIMARY_CONT := Color("#EADDFF")
const C_SURFACE := Color("#FEF7FF")
const C_SURFACE_CONT := Color("#F3EDF7")
const C_ON_SURFACE := Color("#1D1B20")
const C_OUTLINE := Color("#79747E")

const DODGE_KMH := 35.0  # dodge cars: city speed [km/h]
const DODGE_D0 := 2.45  # car half length 2.25 m + 0.2 m: the start distance of tc 0 [m]
var world: Node
var mm: Array = []           # MultiMeshInstance3D per shape kind (capsule, box, sphere)
var cam: Camera3D
var cam_yaw := 0.6
var cam_pitch := 0.25
var cam_dist := 4.0
var cam_target := Vector3(0, 1.0, 0)
var touches := {}            # index -> position
var touch_start := {}        # index -> [position, time]
var pinch_d := 0.0
var tool := 0
var impulse := 60.0
var info: Label
var panel: PanelContainer
var bottom_box: VBoxContainer   # bottom stack (option rows + tool chips); the settings panel ends above it
var sub_row: HBoxContainer
var sub_row2: HBoxContainer   # second option row (foot choice) of the steps tool
var step_foot := 2            # 0 left, 1 right, 2 alternate
var next_foot := 0
var burst := false            # bullet tool: false = one bullet per tap, true = a burst of BURST_N
var burst_left := 0           # bullets of the burst still to fire
var burst_next := 0.0         # sim time of the next bullet
var burst_ray := []           # [from, dir] of the tap: every bullet of the burst goes along the same ray
var shots := 0                # bullets that hit (selftest)
var marker: MeshInstance3D
var marker_t := 0.0
var slow_btn: Button
var pause_btn: Button
var logged := false
var selftest := false
var st_step := 0
var st_t := 0.0
# sizes: 0 de Leva proportions, 1 alternative character proportions; default = the core's, read in _ready
var body_vals := {"height": 1.78, "mass": 80.0, "build": 0.0, "sizes": 0.0}
var BODY_DEFAULTS := {"height": 1.78, "mass": 80.0, "build": 0.0, "sizes": 0.0}
const IMPULSE_DEFAULT := 60.0
var sliders := {}  # title or param name -> HSlider (to show what the core really uses)


func _ready() -> void:
	if not ClassDB.class_exists("EmilWorld"):
		push_error("EMIL: EmilWorld class missing (GDExtension not loaded)")
		return
	world = ClassDB.instantiate("EmilWorld")
	add_child(world)
	BODY_DEFAULTS.sizes = float(world.sizes)  # the core's default body set
	body_vals.sizes = BODY_DEFAULTS.sizes
	if not world.rebuild():
		push_error("EMIL: world build failed")
		return
	print("EMIL: module loaded, bots=%d" % world.bot_count)
	_make_scene()
	_make_ui()
	selftest = "--selftest" in OS.get_cmdline_user_args()


func _make_scene() -> void:
	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color("#E7E0EC")
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.75, 0.75, 0.8)
	e.ambient_light_energy = 0.6
	env.environment = e
	add_child(env)
	var sun := DirectionalLight3D.new()
	sun.rotation = Vector3(deg_to_rad(-55), deg_to_rad(35), 0)
	sun.light_energy = 1.1
	add_child(sun)

	var floor_mi := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(60, 60)
	floor_mi.mesh = pm
	floor_mi.material_override = _mat(Color("#CAC4D0"))
	add_child(floor_mi)
	for i in range(-10, 11):  # grid lines every 1 m
		for axis in 2:
			var l := MeshInstance3D.new()
			var bm := BoxMesh.new()
			bm.size = Vector3(20, 0.002, 0.012) if axis == 0 else Vector3(0.012, 0.002, 20)
			l.mesh = bm
			l.position = Vector3(0, 0.001, i) if axis == 0 else Vector3(i, 0.001, 0)
			l.material_override = _mat(Color("#AEA9B4"))
			add_child(l)

	# Exact physics shapes: capsule = cylinder + 2 spheres; ellipsoid = sphere scaled per axis.
	var meshes := [CylinderMesh.new(), BoxMesh.new(), SphereMesh.new(), SphereMesh.new()]
	# Shaft 2 % thinner than the caps: the cap spheres never coincide with it in depth (no seam, no rings).
	meshes[0].top_radius = 0.49
	meshes[0].bottom_radius = 0.49
	meshes[0].height = 1.0
	meshes[0].radial_segments = 24
	meshes[0].rings = 1
	meshes[0].cap_top = false
	meshes[0].cap_bottom = false
	meshes[1].size = Vector3.ONE
	meshes[2].radius = 0.5
	meshes[2].height = 1.0
	meshes[2].radial_segments = 16
	meshes[2].rings = 8
	# Caps: full spheres (the hemisphere base disc showed as light rings at the joints).
	meshes[3].radius = 0.5
	meshes[3].height = 1.0
	meshes[3].radial_segments = 24
	meshes[3].rings = 12
	# One body colour for every shape kind (mannequin); boxes (feet) a slightly darker shade of it.
	var body := Color("#C8B6D8")
	var cols := [body, body.darkened(0.25), body, body, body, Color("#19191C")]
	# Skull (kind 4) and its drawn face lines (kind 5): triangle meshes from the core, posed like the other shapes.
	for which in 2:
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		for v in world.get_mesh_tris(which):
			st.add_vertex(v)
		st.generate_normals()
		meshes.append(st.commit())
	for k in 6:
		var m := MultiMesh.new()
		m.transform_format = MultiMesh.TRANSFORM_3D
		m.mesh = meshes[k]
		m.instance_count = world.get_count(k)
		var mi := MultiMeshInstance3D.new()
		mi.multimesh = m
		mi.material_override = _mat(cols[k])
		add_child(mi)
		mm.append(mi)

	marker = MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.03
	sm.height = 0.06
	marker.mesh = sm
	marker.material_override = _mat(Color("#B3261E"), true)
	marker.visible = false
	add_child(marker)

	cam = Camera3D.new()
	cam.fov = 55
	add_child(cam)
	_update_cam()


func _mat(c: Color, unshaded := false) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.roughness = 0.8
	if unshaded:
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return m


# ---------- UI (Material 3 look: rounded tonal buttons, surface panels) ----------

func _sb(bg: Color, radius := 24, border := Color(0, 0, 0, 0)) -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = bg
	s.set_corner_radius_all(radius)
	s.content_margin_left = 20
	s.content_margin_right = 20
	s.content_margin_top = 10
	s.content_margin_bottom = 10
	if border.a > 0:
		s.set_border_width_all(2)
		s.border_color = border
	return s


func _theme() -> Theme:
	var t := Theme.new()
	t.default_font_size = 22
	t.set_stylebox("normal", "Button", _sb(C_SURFACE_CONT))
	t.set_stylebox("hover", "Button", _sb(C_PRIMARY_CONT))
	t.set_stylebox("pressed", "Button", _sb(C_PRIMARY))
	t.set_stylebox("hover_pressed", "Button", _sb(C_PRIMARY))
	t.set_stylebox("focus", "Button", StyleBoxEmpty.new())
	t.set_stylebox("disabled", "Button", _sb(Color(C_SURFACE_CONT, 0.5)))
	t.set_color("font_disabled_color", "Button", C_OUTLINE)
	t.set_color("font_color", "Button", C_ON_SURFACE)
	t.set_color("font_hover_color", "Button", C_ON_SURFACE)
	t.set_color("font_pressed_color", "Button", C_ON_PRIMARY)
	t.set_color("font_hover_pressed_color", "Button", C_ON_PRIMARY)
	t.set_color("font_focus_color", "Button", C_ON_SURFACE)
	t.set_stylebox("panel", "PanelContainer", _sb(C_SURFACE, 28))
	t.set_color("font_color", "Label", C_ON_SURFACE)
	var track := _sb(C_PRIMARY_CONT, 6)
	track.content_margin_top = 4
	track.content_margin_bottom = 4
	t.set_stylebox("slider", "HSlider", track)
	var fill := _sb(C_PRIMARY, 6)
	fill.content_margin_top = 4
	fill.content_margin_bottom = 4
	t.set_stylebox("grabber_area", "HSlider", fill)
	t.set_stylebox("grabber_area_highlight", "HSlider", fill)
	return t


func _btn(text: String, toggle := false) -> Button:
	var b := Button.new()
	b.text = text
	b.toggle_mode = toggle
	b.custom_minimum_size = Vector2(0, 64)
	b.focus_mode = Control.FOCUS_NONE
	return b


func _make_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var root := Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.theme = _theme()
	layer.add_child(root)

	info = Label.new()
	info.position = Vector2(16, 12)
	info.add_theme_font_size_override("font_size", 20)
	root.add_child(info)

	# Top right: reset, x0.25, pause, settings.
	var top := HBoxContainer.new()
	top.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	top.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	top.position = Vector2(-16, 12)
	top.add_theme_constant_override("separation", 10)
	root.add_child(top)
	var rb := _btn("Сброс")
	rb.pressed.connect(func(): world.reset())
	top.add_child(rb)
	slow_btn = _btn("×0.25", true)
	slow_btn.toggled.connect(func(on): world.time_scale = 0.25 if on else 1.0)
	top.add_child(slow_btn)
	pause_btn = _btn("Пауза", true)
	pause_btn.toggled.connect(func(on): world.paused = on)
	top.add_child(pause_btn)
	var sb := _btn("Настройки", true)
	sb.toggled.connect(func(on): panel.visible = on)
	top.add_child(sb)

	# Bottom: tool chips (one group) and a sub-option row.
	var bottom := VBoxContainer.new()
	bottom_box = bottom
	bottom.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	bottom.grow_vertical = Control.GROW_DIRECTION_BEGIN
	bottom.offset_left = 16
	bottom.offset_right = -16
	bottom.offset_bottom = -12
	bottom.add_theme_constant_override("separation", 10)
	bottom.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(bottom)
	sub_row2 = HBoxContainer.new()
	sub_row2.alignment = BoxContainer.ALIGNMENT_CENTER
	sub_row2.add_theme_constant_override("separation", 10)
	sub_row2.mouse_filter = Control.MOUSE_FILTER_IGNORE
	bottom.add_child(sub_row2)
	sub_row = HBoxContainer.new()
	sub_row.alignment = BoxContainer.ALIGNMENT_CENTER
	sub_row.add_theme_constant_override("separation", 10)
	sub_row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	bottom.add_child(sub_row)
	var tools := HBoxContainer.new()
	tools.alignment = BoxContainer.ALIGNMENT_CENTER
	tools.add_theme_constant_override("separation", 10)
	bottom.add_child(tools)
	var grp := ButtonGroup.new()
	for i in TOOLS.size():
		var b := _btn(TOOLS[i], true)
		b.button_group = grp
		b.custom_minimum_size = Vector2(130, 72)
		b.button_pressed = i == 0
		b.pressed.connect(_on_tool.bind(i))
		tools.add_child(b)
	_fill_sub()

	# Settings panel on the right, max 1/3 of the screen width.
	panel = PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_RIGHT_WIDE)
	panel.anchor_left = 0.70
	panel.offset_left = 0
	panel.offset_top = 90
	panel.offset_bottom = -190
	panel.offset_right = -16
	panel.visible = false
	root.add_child(panel)
	var sc := ScrollContainer.new()
	sc.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panel.add_child(sc)
	var v := VBoxContainer.new()
	v.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	v.add_theme_constant_override("separation", 6)
	sc.add_child(v)
	var db := _btn("По умолчанию")
	db.pressed.connect(_reset_defaults)
	v.add_child(db)
	_slider(v, "Сила, Н·с", 5, 300, impulse, 1, func(x): impulse = x, "impulse")
	_slider(v, "Рост, м", 1.5, 2.0, body_vals.height, 0.01, func(x): body_vals.height = x, "height")
	_slider(v, "Масса, кг", 50, 120, body_vals.mass, 1, func(x): body_vals.mass = x, "mass")
	_slider(v, "Сложение", -1, 1, body_vals.build, 0.05, func(x): body_vals.build = x, "build")
	_slider(v, "Размеры: 0 де Лева, 1 персонаж", 0, 1, body_vals.sizes, 1, func(x): body_vals.sizes = x, "sizes")
	var ab := _btn("Применить тело")
	ab.pressed.connect(_apply_body)
	v.add_child(ab)
	var plist := {}
	for p in world.get_param_list():
		plist[p.name] = p
	for n in BALANCE_PARAMS:
		if not plist.has(n):
			push_warning("EMIL: no core parameter " + n)
			continue
		var p: Dictionary = plist[n]
		var nm: String = n
		_slider(v, BALANCE_PARAMS[n], p.min, p.max, p.value, 0.0, func(x): world.set_param(nm, x), nm)  # no step: defaults exact


func _slider(parent: Control, title: String, lo: float, hi: float, val: float, step: float, cb: Callable, key := "") -> void:
	var l := Label.new()
	l.add_theme_font_size_override("font_size", 18)
	parent.add_child(l)
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = step
	s.value = val
	s.custom_minimum_size = Vector2(0, 44)
	s.focus_mode = Control.FOCUS_NONE
	var upd := func(x):
		l.text = "%s: %s" % [title, String.num(x, 3)]
		cb.call(x)
	s.value_changed.connect(upd)
	l.text = "%s: %s" % [title, String.num(val, 3)]
	parent.add_child(s)
	if key != "":
		sliders[key] = s


# Rebuild with the new body. Build() resets the core parameters to their defaults, so the user's values are saved
# before and written back after: the sliders keep showing what the core really uses.
func _apply_body() -> void:
	var keep := {}
	for p in world.get_param_list():
		keep[p.name] = p.value
	world.height = body_vals.height
	world.mass = body_vals.mass
	world.build = body_vals.build
	world.sizes = int(body_vals.sizes)
	world.rebuild()
	for n in keep:
		world.set_param(n, keep[n])
	for k in 6:
		mm[k].multimesh.instance_count = world.get_count(k)


# Everything back to the defaults: core parameters (also those without a slider), body, impulse; sliders follow.
func _reset_defaults() -> void:
	for p in world.get_param_list():
		world.set_param(p.name, p.default)
		if sliders.has(p.name):
			sliders[p.name].set_value_no_signal(p.default)
			sliders[p.name].value_changed.emit(p.default)
	impulse = IMPULSE_DEFAULT
	sliders["impulse"].value = IMPULSE_DEFAULT
	var changed := false
	for k in BODY_DEFAULTS:
		if body_vals[k] != BODY_DEFAULTS[k]:
			changed = true
		body_vals[k] = BODY_DEFAULTS[k]
		sliders[k].set_value_no_signal(BODY_DEFAULTS[k])
		sliders[k].value_changed.emit(BODY_DEFAULTS[k])
	if changed:
		_apply_body()


func _on_tool(i: int) -> void:
	tool = i
	if i == 2:
		world.trip(0)
	_fill_sub()


func _fill_sub() -> void:
	for c in sub_row.get_children():
		c.queue_free()
	for c in sub_row2.get_children():
		c.queue_free()
	if tool == 1:
		_fill_bullet()
	elif tool == 3:
		for h in [1.0, 2.0, 4.0]:
			var b := _btn("%d м" % int(h))
			b.pressed.connect(func(): world.drop_from(h))
			sub_row.add_child(b)
	elif tool == 4:
		for spd in [18.0, 50.0]:
			for side in [1, -1]:
				var b := _btn("%d км/ч %s" % [int(spd), "слева" if side > 0 else "справа"])
				b.pressed.connect(func(): world.car(spd, side))
				sub_row.add_child(b)
		# Cars that can be dodged: 35 km/h, start distance from the time to the hit tc (as in bench avoid:
		# car centre 2.25 m half length + 0.2 m body + speed x tc): calm tc 2.0, hurried 1.3, late 0.9
		for side in [0, 1]:
			for d in [["спокойно", 2.0], ["торопливо", 1.3], ["поздно", 0.9]]:
				var tc: float = d[1]
				var start: float = DODGE_D0 + DODGE_KMH / 3.6 * tc
				var b := _btn("%s %s" % ["спереди" if side == 0 else "слева", d[0]])
				b.pressed.connect(func(): world.car(DODGE_KMH, side, start))
				sub_row2.add_child(b)
	elif tool == 2:
		var b := _btn("Ещё подножка")
		b.pressed.connect(func(): world.trip(0))
		sub_row.add_child(b)
	elif tool == TOOL_STEPS:
		_fill_steps()


# ---------- bullet tool: one bullet or a burst per tap, kill ----------

func _fill_bullet() -> void:
	var grp := ButtonGroup.new()
	var names := ["Одна", "Очередь"]
	for i in 2:
		var b := _btn(names[i], true)
		b.button_group = grp
		b.button_pressed = (i == 1) == burst
		b.pressed.connect(func(): burst = i == 1)
		sub_row.add_child(b)
	# "Does not die": health stays (core param immortal); a head bullet still kills
	var ib := _btn("Не умирает", true)
	ib.button_pressed = world.get_param("immortal") > 0.5
	ib.toggled.connect(func(on: bool): world.set_param("immortal", 1.0 if on else 0.0))
	sub_row.add_child(ib)
	var kb := _btn("Убить")
	kb.pressed.connect(func(): world.kill(0))
	sub_row.add_child(kb)


# One bullet along the ray (picked again: the body moves between the bullets of a burst); true if it hit.
func _shoot_ray(from: Vector3, dir: Vector3) -> bool:
	var hit: Dictionary = world.pick(from, dir)
	if hit.is_empty():
		return false
	var p: Vector3 = hit.point
	world.shoot(hit.bot, hit.body, p, Vector3(dir.x, 0, dir.z))
	shots += 1
	marker.position = p
	marker.visible = true
	marker_t = 0.6
	return true


func _start_burst(from: Vector3, dir: Vector3) -> void:
	burst_ray = [from, dir]
	burst_left = BURST_N
	burst_next = world.get_sim_time()
	_burst_tick()


# Fires the due bullets of the burst (sim time: the slow motion slows the burst too; a reset restarts it).
func _burst_tick() -> void:
	if burst_left <= 0:
		return
	var t: float = world.get_sim_time()
	if t + 0.5 < burst_next:  # sim time went back (reset): next bullet now
		burst_next = t
	while burst_left > 0 and t >= burst_next - 1e-6:
		_shoot_ray(burst_ray[0], burst_ray[1])
		burst_left -= 1
		burst_next += BURST_DT


# ---------- steps tool (the step ladder by hand) ----------

func _small_btn(text: String, toggle := false) -> Button:
	var b := _btn(text, toggle)
	b.custom_minimum_size = Vector2(0, 64)  # as the other option buttons (about 39 dp on a 2772x1240 screen)
	return b


func _fill_steps() -> void:
	# Row 2 (upper): which foot.
	var grp := ButtonGroup.new()
	var names := ["Левая нога", "Правая нога", "По очереди"]
	for i in 3:
		var fb := _small_btn(names[i], true)
		fb.button_group = grp
		fb.button_pressed = i == step_foot
		fb.pressed.connect(func(): step_foot = i)
		sub_row2.add_child(fb)
	# Row 1 (lower): actions.
	var wl := _small_btn("Вес на левую")
	wl.pressed.connect(func(): world.shift(0, 0, 0.0))
	sub_row.add_child(wl)
	var wr := _small_btn("Вес на правую")
	wr.pressed.connect(func(): world.shift(0, 1, 0.0))
	sub_row.add_child(wr)
	var one := _small_btn("На одну ногу")
	one.pressed.connect(func(): world.shift(0, _pick_foot(), SINGLE_LEG_S))
	sub_row.add_child(one)
	var acts := [["Шаг вперёд", STEP_FWD, 0.0, false], ["Шаг назад", STEP_BACK, 0.0, false],
			["Шаг вбок", 0.0, STEP_SIDE, false], ["Накрест", STEP_CROSS_DX, STEP_CROSS_DY, true]]
	for a in acts:
		var ab := _small_btn(a[0])
		ab.pressed.connect(func(): world.step(0, _pick_foot(), a[1], a[2], a[3]))
		sub_row.add_child(ab)


# Foot for the next command: the chosen one, or the two feet in turn.
func _pick_foot() -> int:
	if step_foot < 2:
		return step_foot
	var f := next_foot
	next_foot = 1 - next_foot
	return f


# ---------- input: drag = orbit, pinch = zoom, tap on bot = hit point ----------

func _unhandled_input(ev: InputEvent) -> void:
	if world == null:
		return
	if ev is InputEventScreenTouch:
		if ev.pressed:
			touches[ev.index] = ev.position
			touch_start[ev.index] = [ev.position, Time.get_ticks_msec()]
			if touches.size() == 2:
				pinch_d = _pinch()
		else:
			if touch_start.has(ev.index) and touches.size() == 1:
				var st: Array = touch_start[ev.index]
				if ev.position.distance_to(st[0]) < 20 and Time.get_ticks_msec() - st[1] < 400:
					_tap(ev.position)
			touches.erase(ev.index)
			touch_start.erase(ev.index)
	elif ev is InputEventScreenDrag:
		touches[ev.index] = ev.position
		if touches.size() == 1:
			cam_yaw -= ev.relative.x * 0.006
			cam_pitch = clamp(cam_pitch + ev.relative.y * 0.004, -0.1, 1.4)
		elif touches.size() == 2:
			var d := _pinch()
			if pinch_d > 1 and d > 1:
				cam_dist = clamp(cam_dist * pinch_d / d, 1.2, 15.0)
			pinch_d = d
	elif ev is InputEventMouseButton and ev.pressed:
		if ev.button_index == MOUSE_BUTTON_WHEEL_UP:
			cam_dist = max(1.2, cam_dist * 0.9)
		elif ev.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			cam_dist = min(15.0, cam_dist * 1.1)


func _pinch() -> float:
	var k := touches.keys()
	return touches[k[0]].distance_to(touches[k[1]])


func _tap(pos: Vector2) -> void:
	if tool != 0 and tool != 1:
		return
	var from := cam.project_ray_origin(pos)
	var dir := cam.project_ray_normal(pos)
	if tool == 1:
		if burst:
			_start_burst(from, dir)
		else:
			_shoot_ray(from, dir)
		return
	var hit: Dictionary = world.pick(from, dir)
	if hit.is_empty():
		return
	var p: Vector3 = hit.point
	var d := Vector3(dir.x, 0, dir.z)
	# The push acts at the standard point (0.34 m over the hip joints' midpoint); the core param push_old = 1 puts it
	# at the picked point instead
	world.push_std(hit.bot, hit.body, p, d, impulse)
	marker.position = p
	marker.visible = true
	marker_t = 0.6


func _update_cam() -> void:
	var off := Vector3(cos(cam_pitch) * sin(cam_yaw), sin(cam_pitch), cos(cam_pitch) * cos(cam_yaw)) * cam_dist
	cam.position = cam_target + off
	cam.look_at(cam_target, Vector3.UP)


func _process(delta: float) -> void:
	if world == null or cam == null:
		return
	for k in 6:
		if mm[k].multimesh.instance_count > 0:
			mm[k].multimesh.buffer = world.get_buffer(k)
	var bp: Vector3 = world.get_bot_pos(0)
	# The bot is framed above the bottom stack: with the two Steps rows the view looks lower, so the feet stay free.
	var ty := 0.5 if tool == TOOL_STEPS else 0.9
	cam_target = cam_target.lerp(Vector3(bp.x, ty, bp.z), min(1.0, delta * 3.0))
	if panel != null and bottom_box != null:  # settings panel never covers the buttons below it
		var root_h := get_viewport().get_visible_rect().size.y
		panel.offset_bottom = minf(-12.0, bottom_box.get_global_rect().position.y - root_h - 10.0)
	_update_cam()
	_burst_tick()
	if marker_t > 0:
		marker_t -= delta
		marker.visible = marker_t > 0
	if selftest:
		_selftest_tick()
	if not logged and world.get_sim_time() > 3.0:
		logged = true
		print("EMIL: t=%.1f s pelvis_y=%.3f mode=%d core_ms=%.3f" % [world.get_sim_time(), bp.y, world.get_bot_mode(0), world.get_core_ms()])
	info.text = "%d к/с   ядро %.2f мс   t %.1f с   здоровье %d из 100" % [Engine.get_frames_per_second(), world.get_core_ms(),
			world.get_sim_time(), int(world.get_health(0))]


# Headless check of every command (run: godot --headless --path editor --fixed-fps 60 --quit-after 2400 -- --selftest).
func _selftest_tick() -> void:
	st_t += 1.0 / 60.0
	var bp: Vector3 = world.get_bot_pos(0)
	var plan := [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 24.0, 28.0, 30.0, 31.0, 33.0, 36.0]
	if st_step < plan.size() and st_t >= plan[st_step]:
		print("EMIL selftest: step %d t=%.1f pelvis=(%.2f, %.2f, %.2f) mode=%d ms=%.3f" % [st_step, st_t, bp.x, bp.y, bp.z, world.get_bot_mode(0), world.get_core_ms()])
		match st_step:
			0:
				var hit: Dictionary = world.pick(bp + Vector3(0, 0.3, 3), Vector3(0, 0, -1))
				print("EMIL selftest: pick ", hit)
				if not hit.is_empty():
					world.push(hit.bot, hit.body, hit.point, Vector3(0, 0, -1), 60.0)
			1:
				world.reset()
				var hit2: Dictionary = world.pick(bp + Vector3(3, 0.4, 0), Vector3(-1, 0, 0))
				if not hit2.is_empty():
					world.shoot(hit2.bot, hit2.body, hit2.point, Vector3(-1, 0, 0))
			2:
				world.reset()
				world.trip(0)
			3:
				world.reset()
				world.car(18.0, 1)
			4:
				world.drop_from(2.0)
			5:
				world.curl(0)
			6:
				world.reset()
				world.set_param("leg_k", 0.5)
				print("EMIL selftest: leg_k=", world.get_param("leg_k"))
				# Body rebuild keeps the user's parameters; "По умолчанию" restores defaults and the sliders.
				body_vals.height = 1.80
				_apply_body()
				var kept: float = world.get_param("leg_k")
				_reset_defaults()
				var def := 0.0
				for p in world.get_param_list():
					if p.name == "leg_k":
						def = p.default
				var sl: float = sliders["leg_k"].value if sliders.has("leg_k") else def
				var ok := absf(kept - 0.5) < 1e-6 and absf(world.get_param("leg_k") - def) < 1e-6 and absf(sl - def) < 1e-3 \
						and absf(body_vals.height - 1.78) < 1e-6
				print("EMIL selftest: rebuild kept leg_k=%.2f, reset leg_k=%.2f slider=%.2f height=%.2f sliders=%d -> %s" % [kept,
						world.get_param("leg_k"), sl, body_vals.height, sliders.size(), "ok" if ok else "ERROR"])
			7:
				# Steps tool: weight shift, commanded step.
				world.reset()
				world.shift(0, 0, 0.0)
				world.step(0, _pick_foot(), STEP_FWD, 0.0, false)
			8:
				print("EMIL selftest: after step pelvis=(%.2f, %.2f, %.2f) mode=%d steps_ok" % [bp.x, bp.y, bp.z, world.get_bot_mode(0)])
			9:
				world.reset()
			10:
				# Bullet tool, burst: 4 bullets 0.15 s apart along one ray at the chest.
				shots = 0
				_start_burst(bp + Vector3(3, 0.4, 0), Vector3(-1, 0, 0))
			11:
				print("EMIL selftest: burst shots=%d of %d, left=%d -> %s" % [shots, BURST_N, burst_left,
						"ok" if shots == BURST_N and burst_left == 0 else "ERROR"])
				# Kill: the muscles fade, the body falls by itself.
				world.reset()
				world.kill(0)
			12:
				print("EMIL selftest: kill pelvis_y=%.2f -> %s" % [bp.y, "ok" if bp.y < 0.5 else "ERROR"])
		st_step += 1
