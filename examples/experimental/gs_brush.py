"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import colorsys
import random
import copy

import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData
from PIL import Image

import viser
from viser import transforms as tf


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""

def point_to_polyline_distance(points, polyline):
    """
    Compute minimum distance from each point to a polyline (sequence of line segments).
    
    points: (N, 2) array of points
    polyline: (M, 2) array of polyline vertices
    
    Returns:
        min_dists: (N,) array of minimum distances
    """
    min_dists = np.full(points.shape[0], np.inf)
    
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        seg = p2 - p1
        
        v = points - p1  # vectors from p1 to points
        seg_len_sq = np.sum(seg ** 2)
        seg_unit = seg / np.sqrt(seg_len_sq + 1e-8)
        
        # project points onto the line, clamp to segment
        t = np.clip(np.sum(v * seg_unit, axis=1) / (np.sqrt(seg_len_sq) + 1e-8), 0, np.sqrt(seg_len_sq))
        proj = p1 + np.outer(t / (np.sqrt(seg_len_sq) + 1e-8), seg)
        
        dists = np.linalg.norm(points - proj, axis=1)
        min_dists = np.minimum(min_dists, dists)
    
    return min_dists


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def main(splat_paths: tuple[Path, ...] = ()) -> None:
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=False)
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _() -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    if not splat_paths:
        print("Red splat loaded...")
        scale = 0.1
        splat_data: SplatFile = {
            "centers": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "rgbs": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "opacities": np.array([[1.0]], dtype=np.float32),
            "covariances": np.array([np.eye(3, dtype=np.float32) * scale**2], dtype=np.float32),
        }
        for rgb in splat_data["rgbs"]:
            print(rgb)  # This will print something like [1. 0. 0.]
        server.scene.add_transform_controls(f"/0")
        gs_handle = server.scene.add_gaussian_splats(
            f"/0/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )

        remove_button = server.gui.add_button(f"Remove splat object 0")

        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()

        paint_all_button_handle = server.gui.add_button("Paint all splats", icon=viser.Icon.PAINT)
        @paint_all_button_handle.on_click
        def _(_, gs_handle=gs_handle):
            print("Print all pressed!")
            for rgb in splat_data["rgbs"]:
                splat_data["rgbs"][:] = np.array([1.0, 0.0, 1.0])
            for rgb in splat_data["rgbs"]:
                print(rgb)  # This will print something like [1. 0. 0.]
            
            gs_handle.remove()
            gs_handle = server.scene.add_gaussian_splats(
                f"/0/gaussian_splats",
                centers=splat_data["centers"],
                rgbs=splat_data["rgbs"],
                opacities=splat_data["opacities"],
                covariances=splat_data["covariances"],
            )
        paint_selection_button_handle = server.gui.add_button("Paint selection", icon=viser.Icon.PAINT)
        @paint_selection_button_handle.on_click
        def _(_, gs_handle=gs_handle):
            paint_selection_button_handle.disabled = True

            @server.scene.on_pointer_event(event_type="rect-select")
            def _(message: viser.ScenePointerEvent, gs_handle=gs_handle) -> None:
                server.scene.remove_pointer_callback()

                camera = message.client.camera

                # Transform splat centers into the camera frame
                R_camera_world = tf.SE3.from_rotation_and_translation(
                    tf.SO3(camera.wxyz), camera.position
                ).inverse()
                centers_camera_frame = (
                    R_camera_world.as_matrix()
                    @ np.hstack([splat_data["centers"], np.ones((splat_data["centers"].shape[0], 1))]).T
                ).T[:, :3]

                # Project the centers onto the image plane
                fov, aspect = camera.fov, camera.aspect
                centers_proj = centers_camera_frame[:, :2] / centers_camera_frame[:, 2].reshape(-1, 1)
                centers_proj /= np.tan(fov / 2)
                centers_proj[:, 0] /= aspect

                # Normalize to [0, 1] range
                centers_proj = (1 + centers_proj) / 2

                # Check which centers are inside the selected rectangle
                (x0, y0), (x1, y1) = message.screen_pos
                mask = (
                    (centers_proj[:, 0] >= x0) & (centers_proj[:, 0] <= x1) &
                    (centers_proj[:, 1] >= y0) & (centers_proj[:, 1] <= y1)
                )

                # Update the colors of the selected splats
                splat_data["rgbs"][mask] = np.array([0.5, 0.0, 0.7])  # Example color: purple

                # Re-render the splats with updated colors
                gs_handle.remove()
                gs_handle = server.scene.add_gaussian_splats(
                    f"/0/gaussian_splats",
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )
                

            @server.scene.on_pointer_callback_removed
            def _():
                paint_selection_button_handle.disabled = False

    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=True)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")
        
        # keep original splat data to reset back to
        original_splat_data = copy.deepcopy(splat_data)

        # server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )
        
        reset_scene_color_button_handle = server.gui.add_button("Reset scene colors", icon=viser.Icon.RESTORE)
        @reset_scene_color_button_handle.on_click
        def _(_, gs_handle=gs_handle):
            gs_handle.remove()

            splat_data["centers"] = original_splat_data["centers"].copy()
            splat_data["covariances"] = original_splat_data["covariances"].copy()
            splat_data["opacities"] = original_splat_data["opacities"].copy()
            splat_data["rgbs"] = original_splat_data["rgbs"].copy()

            gs_handle = server.scene.add_gaussian_splats(
                    f"/0/gaussian_splats",
                    centers=original_splat_data["centers"],
                    rgbs=original_splat_data["rgbs"],
                    opacities=original_splat_data["opacities"],
                    covariances=original_splat_data["covariances"],
                )

        rgb_button_handle = server.gui.add_rgb("Color picker", (0, 0, 0))

        paint_selection_button_handle = server.gui.add_button("Paint selection", icon=viser.Icon.PAINT)
        @paint_selection_button_handle.on_click
        def _(_, gs_handle=gs_handle, rgb_button_handle=rgb_button_handle):
            paint_selection_button_handle.disabled = True

            @server.scene.on_pointer_event(event_type="rect-select")
            def _(message: viser.ScenePointerEvent, gs_handle=gs_handle, rgb_button_handle=rgb_button_handle) -> None:
                print("Selected color is {0}".format(rgb_button_handle.value))
                server.scene.remove_pointer_callback()

                camera = message.client.camera

                # Code modified from 20_scene_pointer.py
                R_camera_world = tf.SE3.from_rotation_and_translation(
                    tf.SO3(camera.wxyz), camera.position
                ).inverse()
                centers_camera_frame = (
                    R_camera_world.as_matrix()
                    @ np.hstack([splat_data["centers"], np.ones((splat_data["centers"].shape[0], 1))]).T
                ).T[:, :3]

                # Project the centers onto the image plane
                fov, aspect = camera.fov, camera.aspect
                centers_proj = centers_camera_frame[:, :2] / centers_camera_frame[:, 2].reshape(-1, 1)
                centers_proj /= np.tan(fov / 2)
                centers_proj[:, 0] /= aspect

                # Normalize to [0, 1] range
                centers_proj = (1 + centers_proj) / 2

                # create a plane perpendicular to the camera vector
                # get camera position
                v_position = camera.position
                # get camera rotation vector
                forward_camera = np.array([0, 0, -1])
                forward_world = R_camera_world.rotation().apply(forward_camera)
                print("Camera vector is {0}".format(forward_world))

                centers_h = np.hstack([splat_data["centers"], np.ones((splat_data["centers"].shape[0], 1))])

                # Transform centers to camera frame
                centers_camera = (R_camera_world.as_matrix() @ centers_h.T).T[:, :3]  # (N, 3)

                # Select only those with positive z (in front of the camera)
                camera_front_mask = centers_camera[:, 2] > 0
                print("🤖camera_front size {0}".format(camera_front_mask.size))
                print("🐊centers_proj[:, 0] size {0}".format(centers_proj[:, 0].size))

                # Check which centers are inside the selected rectangle
                (x0, y0), (x1, y1) = message.screen_pos
                print("x0 is {0}".format(x0))
                mask = (
                    (centers_proj[:, 0] >= x0) & (centers_proj[:, 0] <= x1) &
                    (centers_proj[:, 1] >= y0) & (centers_proj[:, 1] <= y1) & camera_front_mask
                )   

                # Update colors of select splats
                r = rgb_button_handle.value[0] / 255.0
                g = rgb_button_handle.value[1] / 255.0
                b = rgb_button_handle.value[2] / 255.0
                splat_data["rgbs"][mask] = np.array([r, g, b]) # Paint

                # Update splat by remove and creating new scene with updated colors
                gs_handle.remove()
                gs_handle = server.scene.add_gaussian_splats(
                    f"/0/gaussian_splats",
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )
                

            @server.scene.on_pointer_callback_removed
            def _():
                paint_selection_button_handle.disabled = False
        
        @server.scene.on_pointer_callback_removed
        def _():
                paint_selection_button_handle.disabled = False
        brush_paint_button_handle = server.gui.add_button(
        "Brush paint", 
        icon=viser.Icon.PAINT,
        hint="Paint with a brush tool"
    )

        @brush_paint_button_handle.on_click
        def _(_, gs_handle=gs_handle):
            brush_paint_button_handle.disabled = True
            
            @server.scene.on_pointer_event(event_type="brush")
            def _(message: viser.ScenePointerEvent, gs_handle=gs_handle) -> None:
                server.scene.remove_pointer_callback()
                camera = message.client.camera

                brush_size = 20

                # Transform centers to camera space
                R_camera_world = tf.SE3.from_rotation_and_translation(
                    tf.SO3(camera.wxyz), camera.position
                ).inverse()
                centers_camera_frame = (
                    R_camera_world.as_matrix()
                    @ np.hstack([splat_data["centers"], np.ones((splat_data["centers"].shape[0], 1))]).T
                ).T[:, :3]

                # Only consider points in front of camera
                in_front = centers_camera_frame[:, 2] < 0

                # Project the centers onto the image plane
                fov, aspect = camera.fov, camera.aspect
                centers_proj = centers_camera_frame[:, :2] / centers_camera_frame[:, 2].reshape(-1, 1)
                centers_proj /= np.tan(fov / 2)
                centers_proj[:, 0] /= aspect

                # Normalize to [0, 1] range (origin at bottom-left)
                centers_proj = (1 + centers_proj) / 2

                # Convert brush size from pixels to normalized coordinates
                brush_size_norm = brush_size / 1000.0
                radius = brush_size_norm / 2.0

                # Get brush points
                brush_points = np.array(message.screen_pos)  # (M, 2) array

                # Transform centers to camera frame again (for front mask)
                centers_h = np.hstack([splat_data["centers"], np.ones((splat_data["centers"].shape[0], 1))])
                centers_camera = (R_camera_world.as_matrix() @ centers_h.T).T[:, :3]
                camera_front_mask = centers_camera[:, 2] > 0

                # Compute polyline distance
                min_dists = point_to_polyline_distance(centers_proj, brush_points)
                affected = min_dists <= radius

                # Combine with camera mask
                ported_mask = affected & camera_front_mask

                # Update colors of affected splats
                splat_data["rgbs"][ported_mask] = np.array([1, 0.0, 1])  # magenta

                # Update visualization
                gs_handle.remove()
                gs_handle = server.scene.add_gaussian_splats(
                    f"/0/gaussian_splats",
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )


            @server.scene.on_pointer_callback_removed
            def _():
                brush_paint_button_handle.disabled = False

        global uploaded_image_array
        global uploaded_image_handle
        # Global/shared variables
        uploaded_image_array = None
        uploaded_image_handle = None  # Keep track of the image handle

        upload_button = server.gui.add_upload_button("Upload Image", hint="Upload a JPG or PNG file")

        @upload_button.on_upload
        def _(event: viser.GuiEvent) -> None:
            global uploaded_image_array, uploaded_image_handle

            uploaded_file = upload_button.value
            file_name = uploaded_file.name

            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                with open(file_name, 'wb') as f:
                    f.write(uploaded_file.content)
                print(f"File '{file_name}' uploaded and saved successfully.")

                # Load and store the image as NumPy array (shape H x W x 3)
                image = Image.open(file_name).convert('RGB')
                uploaded_image_array = np.array(image)
                print(f"Image loaded with shape: {uploaded_image_array.shape}")

                # If an old image handle exists, remove it first
                if uploaded_image_handle is not None:
                    uploaded_image_handle.remove()
                    uploaded_image_handle = None

                # Add the new image and store its handle
                uploaded_image_handle = server.gui.add_image(
                    image=uploaded_image_array,
                    label="Uploaded Image",
                    format="jpeg",
                    jpeg_quality=90
                )
            else:
                print("Invalid file type. Please upload a JPG or PNG file.")

        image_selection = server.gui.add_button("Image Selection", icon=viser.Icon.PAINT)
        @image_selection.on_click
        def _(_, gs_handle=gs_handle):
            image_selection.disabled = True

            @server.scene.on_pointer_event(event_type="rect-select")
            def _(message: viser.ScenePointerEvent, gs_handle=gs_handle) -> None:
                server.scene.remove_pointer_callback()

                if uploaded_image_array is None:
                    print("No image uploaded yet.")
                    
                    return

                camera = message.client.camera

                R_camera_world = tf.SE3.from_rotation_and_translation(
                    tf.SO3(camera.wxyz), camera.position
                ).inverse()
                centers_camera_frame = (
                    R_camera_world.as_matrix()
                    @ np.hstack([splat_data["centers"], np.ones((splat_data["centers"].shape[0], 1))]).T
                ).T[:, :3]

                fov, aspect = camera.fov, camera.aspect
                centers_proj = centers_camera_frame[:, :2] / centers_camera_frame[:, 2].reshape(-1, 1)
                centers_proj /= np.tan(fov / 2)
                centers_proj[:, 0] /= aspect

                centers_proj = (1 + centers_proj) / 2

                (x0, y0), (x1, y1) = message.screen_pos
                mask = (
                    (centers_proj[:, 0] >= x0) & (centers_proj[:, 0] <= x1) &
                    (centers_proj[:, 1] >= y0) & (centers_proj[:, 1] <= y1)
                )

                # Normalize within the selection rectangle
                sel_u = (centers_proj[:, 0] - x0) / (x1 - x0 + 1e-8)  # avoid divide by zero
                sel_v = (centers_proj[:, 1] - y0) / (y1 - y0 + 1e-8)

                sel_u = np.clip(sel_u, 0, 1)
                sel_v = np.clip(sel_v, 0, 1)

                # Map to image pixel coordinates
                H, W, _ = uploaded_image_array.shape
                u = np.clip((sel_u * W).astype(int), 0, W - 1)
                v = np.clip((sel_v * H).astype(int), 0, H - 1)

                # Get RGB values from the image and normalize to [0,1]
                image_rgbs = uploaded_image_array[v, u] / 255.0

                # Update only selected splats
                splat_data["rgbs"][mask] = image_rgbs[mask]

                # Re-render the splats
                gs_handle.remove()
                gs_handle = server.scene.add_gaussian_splats(
                    f"/0/gaussian_splats",
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )

            @server.scene.on_pointer_callback_removed
            def _():
                image_selection.disabled = False



    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)