import xatlas
import numpy as np
import time

WATCH_DIR = "XXX"

print("[AMD] Watching for new mesh files...")

while True:
    for file in os.listdir(WATCH_DIR):
        if file.endswith("_vertices.npy"):
            task_id = file.split("_")[0]

            v_path = f"{WATCH_DIR}/{task_id}_vertices.npy"
            f_path = f"{WATCH_DIR}/{task_id}_faces.npy"
            done_flag = f"{WATCH_DIR}/{task_id}.done"

            if os.path.exists(done_flag):
                continue

            if not os.path.exists(f_path):
                continue

            print("[AMD] Processing task", task_id)

            vertices = np.load(v_path)
            faces = np.load(f_path)

            vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
            vertices_new = vertices[vmapping]
            faces_new = indices

            np.save(f"{WATCH_DIR}/{task_id}_vertices_uv.npy", vertices_new)
            np.save(f"{WATCH_DIR}/{task_id}_faces_uv.npy", faces_new)
            np.save(f"{WATCH_DIR}/{task_id}_uvs.npy", uvs)

            with open(done_flag, "w") as f:
                f.write("done")

            print("[AMD] Finished task", task_id)

            os.remove(v_path)
            os.remove(f_path)

    time.sleep(0.5)