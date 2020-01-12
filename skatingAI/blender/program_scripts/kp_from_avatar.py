import bpy
from bpy import context
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from collections import defaultdict
import bmesh
import json


Body = ["DEF-nose",
        "DEF-neck",
        "DEF-deltoid.R",
        "DEF-elbow_fan.R",
        "DEF-palm_index.R",
        "DEF-deltoid.L",
        "DEF-elbow_fan.L",
        "elbow.L",
        "elbow.R",
        "DEF-palm_index.L",
        "DEF-palm_middle.L",
        "DEF-palm_middle.R",
        "DEF-forearm.01.L",
        "DEF-forearm.01.R",
        "DEF-gluteus.R",
        "DEF-knee_fan.R",
        "DEF-foot.R",
        "DEF-gluteus.L",
        "DEF-knee_fan.L",
        "DEF-foot.L",
        "DEF-ear.R",
        "DEF-ear.L", ]


Figure_skating_dress = [
    "DEF-hips",
]
Low_poly = [
    "DEF-eye.R",
    "DEF-eye.L",
]

Ice_skates = ["DEF-nose",
              "DEF-toe.L",
              "DEF-foot.L",
              "DEF-toe.R",
              "DEF-foot.R"]

vertex_group_names = ['Body', 'Figure_skating_dress', 'Low_poly', 'Ice_skates']
vertext_group_obj = [Body, Figure_skating_dress, Low_poly, Ice_skates]


def create_empties(obj_name, group_names):
    ob = bpy.data.objects[f'Figureskater:{obj_name}']
    me = ob.data
    scene = bpy.context.scene
    camera = bpy.data.objects['Camera']
    scene.update()
    print(scene.frame_current)

    keypoints = []

    for name in ob.vertex_groups.keys():
        if name in group_names:

            bpy.ops.object.empty_add(location=(0, 0, 0))
            mt = context.object
            mt.name = f"empty_{ob.name}_{name}"
            cl = mt.constraints.new('COPY_LOCATION')
            cl.target = ob
            cl.subtarget = name

            bpy.context.scene.update()
            mt.matrix_world = mt.matrix_world.copy()
            mt.constraints.clear()

            co_2d = world_to_camera_view(
                bpy.context.scene, bpy.context.scene.camera, mt.location)
            # get pixel coords
            render_scale = scene.render.resolution_percentage / 100
            render_size = (
                int(scene.render.resolution_x * render_scale),
                -int(scene.render.resolution_y * render_scale),
            )

            keypoints.append(
                [co_2d.x * render_size[0], co_2d.y * render_size[1]])

            bpy.ops.object.select_all(action='DESELECT')
            mt.select = True
            bpy.ops.object.delete()

    print(keypoints[0])

    return keypoints


def delete_empties():
    obj = bpy.data.objects

    for ob in obj:
        if 'empty' in ob.name and len(ob.name) > len('empty'):
            print(ob.name)
            bpy.ops.object.select_all(action='DESELECT')
            ob.select = True
            bpy.ops.object.delete()


delete_empties()

allFrames = []
for i in range(context.scene.frame_start, context.scene.frame_end):
    print(i)
    bpy.context.scene.frame_current = i
    bpy.context.scene.frame_set(i)
    print(bpy.context.scene.frame_current)
    bpy.context.scene.update()

    frame = []
    for j, name in enumerate(vertex_group_names):
        print(j, name)
        frame += create_empties(name, vertext_group_obj[j])
    print('*'*100)
    print('frame', i, frame[0])
    allFrames.append(frame)

with open('/home/nadin-katrin/awesome.skating.ai/keypoint_data/keypointsvv4.json', 'w', encoding='utf-8') as f:
    json.dump(allFrames, f, ensure_ascii=False, indent=4)
