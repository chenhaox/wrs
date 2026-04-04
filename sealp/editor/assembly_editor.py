"""
Assembly Sequence Editor — v2.1
================================

Interactive Panda3D-based editor for visualizing and editing assembly
sequences.  Redesigned UI with polished dark theme and console window.

Fixes v2.1:
  - Grab mode uses ray-plane intersection for accurate cursor following
  - Status bar moved below console
  - Button widths corrected to avoid text clipping

Run::

    python -m sealp.editor.run_editor [sequence.yaml]
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

from panda3d.core import (
    TextNode, LPoint3f, LVecBase3f, LVecBase4f,
    CollisionNode, CollisionRay, CollisionTraverser,
    CollisionHandlerQueue, BitMask32, WindowProperties,
    Point3, Vec3, Vec4, Plane, LPoint3,
)
from direct.gui.DirectGui import (
    DirectFrame, DirectLabel, DirectButton,
    DirectScrolledFrame, DirectEntry, DGG,
)

import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.basis.robot_math as rm

from sealp.assembly_sequence import (
    AssemblySequence, AssemblyPart, AssemblyStep,
    save_sequence, load_sequence, SequenceGenerator,
)
from sealp.editor.editor_gui import (
    create_panel, create_section_header, create_label,
    create_entry, create_button, create_accent_button,
    create_list_item, create_separator, ConsoleWindow,
    BG_DARK, BG_MID, BG_LIGHT, ACCENT, ACCENT_BRIGHT, ACCENT_DIM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_ACCENT, TEXT_OK, TEXT_WARN,
    TEXT_SIZE, SMALL_TEXT, TITLE_SIZE, SECTION_SIZE,
    LEFT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN, LIST_BG,
)
from sealp.editor.transform_handler import (
    TransformHandler, TransformMode, AxisConstraint,
)
from sealp.editor.part_manager import PartManager


# ── Layout constants ─────────────────────────────────────────
RIGHT_W = 0.30          # half-width of right panel
LEFT_W = 0.24           # half-width of left panel
CONSOLE_H = 0.26        # height of console bar
STATUS_H = 0.040        # height of status bar
BTN_W = 0.17            # standard button width
BTN_GAP = 0.03          # gap between side-by-side buttons


class AssemblyEditor:
    """Main assembly sequence editor application."""

    def __init__(self, sequence_file: str = None):
        # ── Create world ─────────────────────────────────────
        self.world = wd.World(
            cam_pos=np.array([1.5, -1.5, 1.2]),
            lookat_pos=np.array([0, 0, 0.15]),
            w=1600, h=900,
        )
        self.world.setBackgroundColor(0.22, 0.22, 0.24, 1)

        props = WindowProperties()
        props.setTitle("SEALP Assembly Sequence Editor")
        self.world.win.requestProperties(props)

        # ── Ground grid ──────────────────────────────────────
        self._draw_ground_grid()

        # ── Data ─────────────────────────────────────────────
        self.sequence = AssemblySequence(name="New Assembly")
        self.part_mgr = PartManager(self.world)
        self.transform = TransformHandler(on_mode_change=self._on_mode_change)
        self._selected_step_id: int | None = None
        self._current_file: str | None = sequence_file
        self._grab_plane_z: float = 0.0  # Z height of grab plane

        # ── Build GUI (order: right, left, status, console) ──
        self._build_right_panel()
        self._build_left_panel()
        self._build_status_bar()   # now at the VERY bottom
        self._build_console()      # sits above the status bar

        # ── Key bindings ─────────────────────────────────────
        self._setup_keys()

        # ── Per-frame update ─────────────────────────────────
        self.world.taskMgr.add(self._editor_update, "editor_update")

        # ── Load initial file ────────────────────────────────
        if sequence_file and os.path.isfile(sequence_file):
            self._do_load(sequence_file)
        else:
            self.console.log_info("Ready.  Load a YAML or add parts.")

    # ==============================================================
    # Ground grid
    # ==============================================================
    def _draw_ground_grid(self, size=0.8, step=0.1):
        n = int(size / step)
        for i in range(-n, n + 1):
            v = i * step
            rgb = np.array([0.32, 0.32, 0.34])
            alpha = 0.35
            if i == 0:
                rgb = np.array([0.45, 0.45, 0.48])
                alpha = 0.55
            mgm.gen_stick(spos=np.array([v, -size, 0]),
                          epos=np.array([v, size, 0]),
                          radius=0.0005, rgb=rgb,
                          alpha=alpha).attach_to(self.world)
            mgm.gen_stick(spos=np.array([-size, v, 0]),
                          epos=np.array([size, v, 0]),
                          radius=0.0005, rgb=rgb,
                          alpha=alpha).attach_to(self.world)
        mgm.gen_frame(ax_length=0.12, ax_radius=0.002).attach_to(self.world)

    # ==============================================================
    # Ray-plane intersection helper
    # ==============================================================
    def _mouse_to_world_on_plane(self, plane_z=0.0):
        """Cast a ray from the camera through the mouse position and
        intersect it with the horizontal plane at ``z=plane_z``.

        Returns ``np.array([x, y, z])`` or None.
        """
        if not self.world.mouseWatcherNode.hasMouse():
            return None
        mpos = self.world.mouseWatcherNode.getMouse()

        # Get near and far points on the mouse ray
        near_point = Point3()
        far_point = Point3()
        self.world.camLens.extrude(mpos, near_point, far_point)

        # Transform to world space
        near_world = self.world.render.getRelativePoint(
            self.world.cam, near_point)
        far_world = self.world.render.getRelativePoint(
            self.world.cam, far_point)

        # Intersect with Z=plane_z horizontal plane
        dz = far_world.getZ() - near_world.getZ()
        if abs(dz) < 1e-9:
            return None  # ray parallel to plane
        t = (plane_z - near_world.getZ()) / dz
        if t < 0:
            return None  # plane behind camera

        x = near_world.getX() + t * (far_world.getX() - near_world.getX())
        y = near_world.getY() + t * (far_world.getY() - near_world.getY())
        return np.array([x, y, plane_z])

    # ==============================================================
    # RIGHT PANEL
    # ==============================================================
    def _build_right_panel(self):
        pw = RIGHT_W
        panel_h = 1.92
        self._right_frame = create_panel(
            self.world.a2dTopRight, pw * 2, panel_h,
            (-pw * 2, 0), bg=BG_MID,
        )

        y = -TOP_MARGIN * 2
        inner_w = pw * 2 - LEFT_MARGIN * 2

        # ── Title ────────────────────────────────────────────
        create_label(self._right_frame, "SEALP Editor",
                     (LEFT_MARGIN, y), scale=TITLE_SIZE,
                     color=ACCENT_BRIGHT)
        y -= TITLE_SIZE * 1.6

        # ── PARTS section ────────────────────────────────────
        create_section_header(self._right_frame, "Parts",
                              (LEFT_MARGIN, y), width=inner_w)
        y -= SECTION_SIZE * 1.2

        scroll_h = 0.38
        self._parts_scroll = DirectScrolledFrame(
            canvasSize=(0, inner_w, -0.01, scroll_h),
            frameSize=(0, inner_w, -scroll_h, 0),
            frameColor=LIST_BG,
            pos=LPoint3f(LEFT_MARGIN, 0, y),
            parent=self._right_frame,
            scrollBarWidth=0.015,
            autoHideScrollBars=True,
        )
        self._parts_items = []
        y -= scroll_h + TOP_MARGIN

        # buttons — wider to fit text
        x0 = LEFT_MARGIN + BTN_W / 2
        create_accent_button(self._right_frame, "Load Part",
                             (x0, y), self._on_load_part, width=BTN_W)
        create_button(self._right_frame, "Remove",
                      (x0 + BTN_W + BTN_GAP, y),
                      self._on_remove_part, width=BTN_W)
        y -= TEXT_SIZE * 2.4

        # ── STEPS section ────────────────────────────────────
        create_section_header(self._right_frame, "Assembly Steps",
                              (LEFT_MARGIN, y), width=inner_w)
        y -= SECTION_SIZE * 1.2

        scroll_h2 = 0.30
        self._steps_scroll = DirectScrolledFrame(
            canvasSize=(0, inner_w, -0.01, scroll_h2),
            frameSize=(0, inner_w, -scroll_h2, 0),
            frameColor=LIST_BG,
            pos=LPoint3f(LEFT_MARGIN, 0, y),
            parent=self._right_frame,
            scrollBarWidth=0.015,
            autoHideScrollBars=True,
        )
        self._steps_items = []
        y -= scroll_h2 + TOP_MARGIN

        create_accent_button(self._right_frame, "Add Step",
                             (x0, y), self._on_add_step, width=BTN_W)
        create_button(self._right_frame, "Del Step",
                      (x0 + BTN_W + BTN_GAP, y),
                      self._on_del_step, width=BTN_W)
        y -= TEXT_SIZE * 2.4

        # ── FILE I/O ─────────────────────────────────────────
        create_separator(self._right_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2

        create_accent_button(self._right_frame, "Load YAML",
                             (x0, y), self._on_load_yaml, width=BTN_W)
        create_accent_button(self._right_frame, "Save YAML",
                             (x0 + BTN_W + BTN_GAP, y),
                             self._on_save_yaml, width=BTN_W)
        y -= TEXT_SIZE * 2.4
        create_button(self._right_frame, "New",
                      (x0, y), self._on_new_sequence, width=BTN_W)

    # ==============================================================
    # LEFT PANEL (Properties)
    # ==============================================================
    def _build_left_panel(self):
        pw = LEFT_W
        panel_h = 1.20
        self._left_frame = create_panel(
            self.world.a2dTopLeft, pw * 2, panel_h,
            (0, 0), bg=BG_MID,
        )

        y = -TOP_MARGIN * 2
        inner_w = pw * 2 - LEFT_MARGIN * 2

        create_section_header(self._left_frame, "Properties",
                              (LEFT_MARGIN, y), width=inner_w)
        y -= SECTION_SIZE * 1.5

        # part name
        create_label(self._left_frame, "Part:", (LEFT_MARGIN, y),
                     scale=SMALL_TEXT, color=TEXT_SECONDARY)
        self._prop_name = create_label(
            self._left_frame, "(none)", (0.065, y), color=TEXT_PRIMARY)
        y -= TEXT_SIZE * 1.5

        # model
        create_label(self._left_frame, "Model:", (LEFT_MARGIN, y),
                     scale=SMALL_TEXT, color=TEXT_SECONDARY)
        self._prop_model = create_label(
            self._left_frame, "-", (0.075, y),
            scale=SMALL_TEXT * 0.85, color=TEXT_SECONDARY)
        y -= TEXT_SIZE * 1.8

        # ── Position ─────────────────────────────────────────
        create_separator(self._left_frame, (LEFT_MARGIN, y), inner_w)
        y -= TOP_MARGIN * 2
        create_section_header(self._left_frame, "Position",
                              (LEFT_MARGIN, y), width=inner_w * 0.6,
                              color=ACCENT_BRIGHT)
        y -= SECTION_SIZE * 1.1

        self._pos_entries = {}
        for ax, col in [("X", (0.95, 0.30, 0.30, 1)),
                        ("Y", (0.30, 0.85, 0.30, 1)),
                        ("Z", (0.30, 0.55, 0.95, 1))]:
            create_label(self._left_frame, f"{ax}:", (LEFT_MARGIN, y),
                         scale=SMALL_TEXT, color=col)
            self._pos_entries[ax] = create_entry(
                self._left_frame, (0.05, y), width=0.14, initial="0.000")
            y -= TEXT_SIZE * 1.4
        y -= TOP_MARGIN

        # ── Rotation ─────────────────────────────────────────
        create_section_header(self._left_frame, "Rotation (deg)",
                              (LEFT_MARGIN, y), width=inner_w * 0.6,
                              color=ACCENT_BRIGHT)
        y -= SECTION_SIZE * 1.1

        self._rot_entries = {}
        for ax, col in [("Rx", (0.95, 0.30, 0.30, 1)),
                        ("Ry", (0.30, 0.85, 0.30, 1)),
                        ("Rz", (0.30, 0.55, 0.95, 1))]:
            create_label(self._left_frame, f"{ax}:", (LEFT_MARGIN, y),
                         scale=SMALL_TEXT, color=col)
            self._rot_entries[ax] = create_entry(
                self._left_frame, (0.05, y), width=0.14, initial="0.000")
            y -= TEXT_SIZE * 1.4
        y -= TOP_MARGIN

        # apply button
        create_accent_button(self._left_frame, "Apply",
                             (LEFT_MARGIN + 0.06, y),
                             self._on_apply_properties, width=0.12)
        y -= TEXT_SIZE * 2.2

        # mass
        create_label(self._left_frame, "Mass:", (LEFT_MARGIN, y),
                     scale=SMALL_TEXT, color=TEXT_SECONDARY)
        self._prop_mass = create_label(
            self._left_frame, "0.0 kg", (0.08, y), color=TEXT_PRIMARY)

    # ==============================================================
    # STATUS BAR — at the very bottom
    # ==============================================================
    def _build_status_bar(self):
        self._status_frame = DirectFrame(
            frameSize=(-2, 2, 0, STATUS_H),
            frameColor=BG_DARK,
            pos=LPoint3f(0, 0, 0),         # very bottom
            parent=self.world.a2dBottomCenter,
        )
        self._mode_dot = DirectFrame(
            frameSize=(-0.007, 0.007, -0.007, 0.007),
            frameColor=TEXT_OK,
            pos=LPoint3f(-1.74, 0, STATUS_H * 0.45),
            parent=self._status_frame,
        )
        self._status_label = DirectLabel(
            text="  Mode: SELECT | No part selected",
            text_scale=SMALL_TEXT * 0.88,
            text_fg=TEXT_SECONDARY,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            pos=LPoint3f(-1.72, 0, STATUS_H * 0.22),
            parent=self._status_frame,
        )
        self._hint_label = DirectLabel(
            text="G:Grab  R:Rotate  X/Y/Z:Axis  Esc:Cancel  Ctrl+S:Save",
            text_scale=SMALL_TEXT * 0.78,
            text_fg=(*TEXT_SECONDARY[:3], 0.45),
            text_align=TextNode.ARight,
            frameColor=(0, 0, 0, 0),
            pos=LPoint3f(1.72, 0, STATUS_H * 0.22),
            parent=self._status_frame,
        )

    # ==============================================================
    # CONSOLE — sits above the status bar
    # ==============================================================
    def _build_console(self):
        self.console = ConsoleWindow(
            self.world.a2dBottomCenter,
            width=1.0,
            height=CONSOLE_H,
            y_offset=STATUS_H,   # push up by the status bar height
            max_lines=80,
        )

    # ==============================================================
    # Key bindings
    # ==============================================================
    def _setup_keys(self):
        w = self.world
        w.accept("g", self._key_grab)
        w.accept("r", self._key_rotate)
        w.accept("x", self._key_axis_x)
        w.accept("y", self._key_axis_y)
        w.accept("z", self._key_axis_z)
        w.accept("escape", self._key_escape)
        w.accept("mouse1", self._key_confirm_or_pick)
        w.accept("delete", self._on_remove_part)
        w.accept("control-s", self._on_save_yaml)
        w.accept("control-o", self._on_load_yaml)

    # ==============================================================
    # Key handlers
    # ==============================================================
    def _key_grab(self):
        entry = self.part_mgr.get_selected_entry()
        if entry and not self.transform.active:
            # determine grab plane Z from the part's current height
            self._grab_plane_z = float(entry.cmodel.pos[2])
            world_hit = self._mouse_to_world_on_plane(self._grab_plane_z)
            if world_hit is not None:
                self.transform.start_grab(
                    entry.cmodel.pos.copy(),
                    entry.cmodel.rotmat.copy(),
                    world_hit,
                )
                self.console.log(f"Grab mode: {self.part_mgr.selected_id}",
                                 color=ACCENT_BRIGHT)

    def _key_rotate(self):
        if self.transform.active and self.transform.mode == TransformMode.GRAB:
            self.transform.cancel()
        entry = self.part_mgr.get_selected_entry()
        if entry and not self.transform.active:
            mouse_xy = self._get_mouse_xy()
            if mouse_xy:
                self.transform.start_rotate(
                    entry.cmodel.pos.copy(),
                    entry.cmodel.rotmat.copy(),
                    mouse_xy,
                )
                self.console.log(f"Rotate mode: {self.part_mgr.selected_id}",
                                 color=ACCENT_BRIGHT)

    def _key_axis_x(self):
        if self.transform.active:
            self.transform.constrain(AxisConstraint.X)
            self.console.log_info("Axis constrained to X")

    def _key_axis_y(self):
        if self.transform.active:
            self.transform.constrain(AxisConstraint.Y)
            self.console.log_info("Axis constrained to Y")

    def _key_axis_z(self):
        if self.transform.active:
            # for Z grab: switch to vertical plane
            if self.transform.mode == TransformMode.GRAB:
                self._grab_plane_z = None  # signal to use vertical mode
            self.transform.constrain(AxisConstraint.Z)
            self.console.log_info("Axis constrained to Z")

    def _key_escape(self):
        if self.transform.active:
            pos, rotmat = self.transform.cancel()
            sel = self.part_mgr.selected_id
            if sel:
                self.part_mgr.update_part_pose(sel, pos, rotmat)
                self._refresh_properties()
            self.console.log_warn("Transform cancelled.")
        else:
            self.part_mgr.deselect_all()
            self._refresh_properties()
            self._update_status()

    def _key_confirm_or_pick(self):
        if self.transform.active:
            pos, rotmat = self.transform.confirm()
            sel = self.part_mgr.selected_id
            if sel:
                self.part_mgr.update_part_pose(sel, pos, rotmat)
                self._refresh_properties()
                self.console.log_ok(
                    f"Applied to {sel}: "
                    f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        self._update_status()

    # ==============================================================
    # Per-frame update
    # ==============================================================
    def _editor_update(self, task):
        if self.transform.active:
            sel = self.part_mgr.selected_id
            if self.transform.mode == TransformMode.GRAB:
                # For Z-axis constraint, use mouse Y as a direct offset
                if self.transform.axis == AxisConstraint.Z:
                    mouse_xy = self._get_mouse_xy()
                    if mouse_xy and sel:
                        # use vertical mouse movement for Z
                        dy = mouse_xy[1] - (self.transform._start_mouse[1]
                                            if self.transform._start_mouse
                                            else mouse_xy[1])
                        # create a synthetic world hit
                        new_z = self.transform._origin_pos[2] + dy * 0.5
                        world_hit = self.transform._start_world_pos.copy()
                        world_hit[2] = new_z
                        pos, rotmat = self.transform.update_grab(world_hit)
                        self.part_mgr.update_part_pose(sel, pos, rotmat)
                        self._refresh_properties()
                else:
                    # Normal XY / X / Y grab via ray-plane intersection
                    world_hit = self._mouse_to_world_on_plane(
                        self._grab_plane_z)
                    if world_hit is not None and sel:
                        pos, rotmat = self.transform.update_grab(world_hit)
                        self.part_mgr.update_part_pose(sel, pos, rotmat)
                        self._refresh_properties()
            elif self.transform.mode == TransformMode.ROTATE:
                mouse_xy = self._get_mouse_xy()
                if mouse_xy and sel:
                    pos, rotmat = self.transform.update_rotate(mouse_xy)
                    self.part_mgr.update_part_pose(sel, pos, rotmat)
                    self._refresh_properties()
        return task.cont

    # ==============================================================
    # Parts callbacks
    # ==============================================================
    def _on_load_part(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title="Load 3D Part",
                filetypes=[("3D Models", "*.stl *.obj *.dae"),
                           ("All files", "*.*")],
            )
            root.destroy()
            if filepath:
                self._add_part_from_file(filepath)
        except Exception as e:
            self.console.log_warn(f"File dialog error: {e}")

    def _add_part_from_file(self, filepath: str):
        stem = Path(filepath).stem
        part_id = stem
        idx = 1
        while part_id in [p.part_id for p in self.sequence.parts]:
            part_id = f"{stem}_{idx}"
            idx += 1
        part = AssemblyPart(
            part_id=part_id,
            name=stem.replace("_", " ").title(),
            model_path=filepath,
            init_pos=np.array([0.0, 0.0, 0.0]),
            init_rotmat=np.eye(3),
            assembly_pos=np.array([0.0, 0.0, 0.0]),
            assembly_rotmat=np.eye(3),
        )
        self.sequence.add_part(part)
        self.part_mgr.load_part(part)
        self._refresh_parts_list()
        self._select_part(part_id)
        self.console.log_ok(f"Loaded part: {part_id}")

    def _on_remove_part(self):
        sel = self.part_mgr.selected_id
        if sel:
            self.part_mgr.remove_part(sel)
            if sel in self.sequence._parts:
                del self.sequence._parts[sel]
            self.sequence._steps = [
                s for s in self.sequence._steps if s.part_id != sel
            ]
            self._refresh_parts_list()
            self._refresh_steps_list()
            self._refresh_properties()
            self._update_status()
            self.console.log_warn(f"Removed part: {sel}")

    def _select_part(self, part_id):
        self.part_mgr.select_part(part_id)
        self._refresh_properties()
        self._refresh_parts_list()
        self._update_status()

    # ==============================================================
    # Steps callbacks
    # ==============================================================
    def _on_add_step(self):
        sel = self.part_mgr.selected_id
        if not sel:
            self.console.log_warn("Select a part first to add a step.")
            return
        existing_ids = {s.step_id for s in self.sequence.steps}
        new_id = 0
        while new_id in existing_ids:
            new_id += 1
        deps = sorted(existing_ids) if existing_ids else []
        entry = self.part_mgr.entries.get(sel)
        step = AssemblyStep(
            step_id=new_id,
            part_id=sel,
            parent_part_id="fixture",
            assembly_pos=(entry.cmodel.pos.copy()
                          if entry and entry.cmodel else np.zeros(3)),
            assembly_rotmat=(entry.cmodel.rotmat.copy()
                             if entry and entry.cmodel else np.eye(3)),
            dependencies=deps[-1:] if deps else [],
        )
        self.sequence.add_step(step)
        self._selected_step_id = new_id
        self._refresh_steps_list()
        self.console.log_ok(f"Added step {new_id} for part '{sel}'")

    def _on_del_step(self):
        if self._selected_step_id is not None:
            sid = self._selected_step_id
            self.sequence._steps = [
                s for s in self.sequence._steps if s.step_id != sid
            ]
            self._selected_step_id = None
            self._refresh_steps_list()
            self.console.log_warn(f"Deleted step {sid}")

    def _select_step(self, step_id):
        self._selected_step_id = step_id
        for s in self.sequence.steps:
            if s.step_id == step_id:
                self._select_part(s.part_id)
                self.part_mgr.show_assembly_ghost(s.part_id)
                break
        self._refresh_steps_list()

    # ==============================================================
    # File I/O
    # ==============================================================
    def _on_load_yaml(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title="Load Assembly Sequence",
                filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
            )
            root.destroy()
            if filepath:
                self._do_load(filepath)
        except Exception as e:
            self.console.log_warn(f"Load error: {e}")

    def _do_load(self, filepath: str):
        try:
            self.sequence = load_sequence(filepath)
            self._current_file = filepath
        except Exception as e:
            self.console.log_warn(f"Failed to load {filepath}: {e}")
            return
        for pid in list(self.part_mgr.part_ids):
            self.part_mgr.remove_part(pid)
        for part in self.sequence.parts:
            self.part_mgr.load_part(part)
        self._refresh_parts_list()
        self._refresh_steps_list()
        self._refresh_properties()
        self._update_status()
        self.console.log_ok(
            f"Loaded: {self.sequence.name} "
            f"({self.sequence.n_parts} parts, {self.sequence.n_steps} steps)")

    def _on_save_yaml(self):
        self.part_mgr.sync_to_sequence(self.sequence)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.asksaveasfilename(
                title="Save Assembly Sequence",
                defaultextension=".yaml",
                filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
                initialfile=self._current_file or "assembly.yaml",
            )
            root.destroy()
            if filepath:
                save_sequence(self.sequence, filepath)
                self._current_file = filepath
                self.console.log_ok(f"Saved to {filepath}")
        except Exception as e:
            self.console.log_warn(f"Save error: {e}")

    def _on_new_sequence(self):
        for pid in list(self.part_mgr.part_ids):
            self.part_mgr.remove_part(pid)
        self.sequence = AssemblySequence(name="New Assembly")
        self._current_file = None
        self._selected_step_id = None
        self._refresh_parts_list()
        self._refresh_steps_list()
        self._refresh_properties()
        self._update_status()
        self.console.log_info("New empty sequence created.")

    # ==============================================================
    # Properties panel
    # ==============================================================
    def _refresh_properties(self):
        entry = self.part_mgr.get_selected_entry()
        if entry is None:
            self._prop_name["text"] = "(none)"
            self._prop_model["text"] = "-"
            self._prop_mass["text"] = "0.0 kg"
            for e in self._pos_entries.values():
                e.set("0.000")
            for e in self._rot_entries.values():
                e.set("0.000")
            return
        self._prop_name["text"] = entry.part.name
        self._prop_model["text"] = os.path.basename(entry.part.model_path)
        self._prop_mass["text"] = f"{entry.part.mass:.2f} kg"
        pos = entry.cmodel.pos if entry.cmodel else entry.part.init_pos
        self._pos_entries["X"].set(f"{pos[0]:.4f}")
        self._pos_entries["Y"].set(f"{pos[1]:.4f}")
        self._pos_entries["Z"].set(f"{pos[2]:.4f}")
        rotmat = entry.cmodel.rotmat if entry.cmodel else entry.part.init_rotmat
        try:
            rz = np.arctan2(rotmat[1, 0], rotmat[0, 0])
            ry = np.arctan2(-rotmat[2, 0],
                            np.sqrt(rotmat[2, 1] ** 2 + rotmat[2, 2] ** 2))
            rx = np.arctan2(rotmat[2, 1], rotmat[2, 2])
        except Exception:
            rx = ry = rz = 0.0
        self._rot_entries["Rx"].set(f"{np.degrees(rx):.2f}")
        self._rot_entries["Ry"].set(f"{np.degrees(ry):.2f}")
        self._rot_entries["Rz"].set(f"{np.degrees(rz):.2f}")

    def _on_apply_properties(self):
        sel = self.part_mgr.selected_id
        if not sel:
            return
        try:
            pos = np.array([
                float(self._pos_entries["X"].get()),
                float(self._pos_entries["Y"].get()),
                float(self._pos_entries["Z"].get()),
            ])
            rx = np.radians(float(self._rot_entries["Rx"].get()))
            ry = np.radians(float(self._rot_entries["Ry"].get()))
            rz = np.radians(float(self._rot_entries["Rz"].get()))
        except ValueError:
            self.console.log_warn("Invalid numeric input in properties.")
            return
        rotmat = rm.rotmat_from_euler(rx, ry, rz)
        self.part_mgr.update_part_pose(sel, pos, rotmat)
        self._refresh_properties()
        self.console.log_ok(
            f"Applied pose to {sel}: "
            f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # ==============================================================
    # List refresh
    # ==============================================================
    def _refresh_parts_list(self):
        canvas = self._parts_scroll.getCanvas()
        for item in self._parts_items:
            item.destroy()
        self._parts_items.clear()
        y = -SMALL_TEXT * 0.5
        selected = self.part_mgr.selected_id
        inner_w = RIGHT_W - LEFT_MARGIN
        for pid in self.part_mgr.part_ids:
            entry = self.part_mgr.entries[pid]
            is_sel = (pid == selected)
            btn = create_list_item(
                canvas, f"  {entry.part.name} ({pid})",
                (0, y), width=inner_w,
                on_click=self._select_part,
                item_id=pid, selected=is_sel,
            )
            self._parts_items.append(btn)
            y -= SMALL_TEXT * 1.8
        total_h = max(0.38, abs(y) + SMALL_TEXT)
        inner_w_full = RIGHT_W * 2 - LEFT_MARGIN * 2
        self._parts_scroll["canvasSize"] = (0, inner_w_full, -total_h, 0)

    def _refresh_steps_list(self):
        canvas = self._steps_scroll.getCanvas()
        for item in self._steps_items:
            item.destroy()
        self._steps_items.clear()
        y = -SMALL_TEXT * 0.5
        try:
            ordered = self.sequence.get_execution_order()
        except Exception:
            ordered = self.sequence.steps
        inner_w = RIGHT_W - LEFT_MARGIN
        for step in ordered:
            is_sel = (step.step_id == self._selected_step_id)
            deps = ",".join(str(d) for d in step.dependencies) or "-"
            label = (f"  {step.step_id}: {step.part_id} -> "
                     f"{step.parent_part_id}  [deps:{deps}]")
            btn = create_list_item(
                canvas, label, (0, y), width=inner_w,
                on_click=self._select_step,
                item_id=step.step_id, selected=is_sel,
            )
            self._steps_items.append(btn)
            y -= SMALL_TEXT * 1.8
        total_h = max(0.30, abs(y) + SMALL_TEXT)
        inner_w_full = RIGHT_W * 2 - LEFT_MARGIN * 2
        self._steps_scroll["canvasSize"] = (0, inner_w_full, -total_h, 0)

    # ==============================================================
    # Status bar
    # ==============================================================
    def _update_status(self):
        mode_str = self.transform.mode.name
        sel = self.part_mgr.selected_id or "(none)"
        step_str = (str(self._selected_step_id)
                    if self._selected_step_id is not None else "-")
        if self.transform.active:
            axis_str = self.transform.axis.name
            self._status_label["text"] = (
                f"  Mode: {mode_str} [{axis_str}] | "
                f"Part: {sel} | Step: {step_str}")
            self._mode_dot["frameColor"] = TEXT_WARN
        else:
            self._status_label["text"] = (
                f"  Mode: SELECT | Part: {sel} | Step: {step_str}")
            self._mode_dot["frameColor"] = TEXT_OK

    def _on_mode_change(self, mode, axis):
        self._update_status()

    # ==============================================================
    # Utilities
    # ==============================================================
    def _get_mouse_xy(self):
        if self.world.mouseWatcherNode.hasMouse():
            mp = self.world.mouseWatcherNode.getMouse()
            return (mp.getX(), mp.getY())
        return None

    def run(self):
        self.world.run()
