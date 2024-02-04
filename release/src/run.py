#general imports
import os
import torch
from tqdm import trange
import numpy as np
#paper code
from large_steps.util.load_xml import load_scene
from large_steps.util.render import NVDRenderer
from large_steps.math.optimize import AdamUniform

#own code
from large_steps.own.utils import to_differential,from_differential,system_matrix,face_normals,vertex_normals,to_numpy, save_data, read_cfg, triangles_to_quads, quads_to_triangles, write_leftovers



#-----------------------------------------
#----------------- SETUP -----------------
print("Running: SETUP")


filepath = os.path.join(os.getcwd(), "scenes","suzanne","suzanne.xml")
scene_params = load_scene(filepath)

v_ref = scene_params["mesh-target"]["vertices"]
n_ref = scene_params["mesh-target"]["normals"]
f_ref = scene_params["mesh-target"]["faces"]

v = scene_params["mesh-source"]["vertices"]
f = scene_params["mesh-source"]["faces"]

#----------------- SETUP -----------------
#-----------------------------------------

#------------------------------------------
#----------------- RENDER -----------------
print("Running: RENDER")
renderer = NVDRenderer(scene_params,shading=True,boost=3)
imgs = renderer.render(v_ref,n_ref,f_ref)
#----------------- RENDER -----------------
#------------------------------------------

#-----------------------------------------------
#----------------- PARAMETRIZE -----------------
print("Running: PARAMETRIZE")
cfg_params = read_cfg()
save_path = cfg_params[0]
iterations = int(cfg_params[1])
step_size = float(cfg_params[2])
lambda_= int(cfg_params[3])
M = system_matrix(v,f,lambda_)
u = to_differential(M,v)
u.requires_grad = True
opt = AdamUniform([u],step_size)

v_its = torch.zeros((iterations+1,*v.shape),device='cuda')
losses = torch.zeros(iterations+1,device='cuda')
#----------------- PARAMETRIZE -----------------
#-----------------------------------------------

#--------------------------------------------
#----------------- OPTIMIZE -----------------
print("Running: OPTIMIZE")
for i in trange(0,iterations):
    v = from_differential(M,u)
    f_normals = face_normals(v,f)
    v_normals = vertex_normals(v,f,f_normals)

    #f_quad, f_tri = tris_to_quads(f)
    #new_imgs = renderer.render(v,v_normals,f_quad,f_tri)

    new_imgs = renderer.render(v,v_normals,f)
    loss = (new_imgs - imgs).abs().mean()

    #f_old = tris_to_quads(f_quad)
    #f = torch.cat(f_tri,f_old) 

    with torch.no_grad():
        losses[i] = loss
        v_its[i] = v
    
    opt.zero_grad()
    loss.backward()
    opt.step()

#----------------- OPTIMIZE -----------------
#--------------------------------------------

#----------------------------------------------
#------------------- OUTPUT -------------------
print("Running: OUTPUT")
with torch.no_grad():
    opt_imgs = renderer.render(v,v_normals,f)
    loss = (new_imgs - imgs).abs().mean()
    losses[-1] = loss
    v_its[-1] = v


numpy_f_normals = to_numpy(face_normals(v_its[-1],f)).transpose()
numpy_v = to_numpy(v_its[-1])
f_quads, f_tris = triangles_to_quads(f)
numpy_f_quads = to_numpy(f_quads)
numpy_f_tris = to_numpy(f_tris)


save_data(numpy_v,numpy_f_quads,save_path)
write_leftovers(numpy_f_tris,save_path)

#------------------- OUTPUT -------------------   
#----------------------------------------------