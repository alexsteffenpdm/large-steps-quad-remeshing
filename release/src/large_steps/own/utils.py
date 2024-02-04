import numpy as np
import torch
import os
from tqdm import trange
from large_steps.util.solvers import CholeskySolver, solve
from large_steps.util.io_ply import write_ply
import weakref

# Cache for the system solvers
_cache = {}

def cache_put(key, value, A):
    # Called when 'A' is garbage collected
    def cleanup_callback(wr):
        del _cache[key]

    wr = weakref.ref(
        A,
        cleanup_callback
    )

    _cache[key] = (value, wr)

def get_corresponding(v0,v2,faces,converted):
    for i in range(faces.shape[0]):
        if faces[i,0]== v0 and faces[i,1]==v2 and converted[i] == False:
            return i
    return -1


def triangles_to_quads(faces): 
    quads = []
    faces = faces.cpu().numpy()
   
    converted = np.full(faces.shape[0],False)
    for i in trange(faces.shape[0]):
        if converted[i] == False:
            v0 = faces[i,0]
            v1 = faces[i,1]
            v2 = faces[i,2]
            corr_index = get_corresponding(v0,v2,faces,converted)
            if corr_index != -1:
                v3 = faces[corr_index][2]
                quad = [v0,v1,v2,v3]
                converted[i] = True
                converted[corr_index] = True
                quads.append(quad)   
    leftovers = []
    for i,c in enumerate(converted):
        if c == False:
            leftovers.append(faces[i])


    return torch.from_numpy(np.asarray(quads)),torch.from_numpy(np.asarray(leftovers))

def quads_to_triangles(faces):
    triangles = []
    for f in faces:
        t1 = [f[0],f[1],f[2]].cpu().numpy()
        t2 = [f[0],f[2],f[3]].cpu().numpy()
        triangles.append(t1)
        triangles.append(t2)
    return torch.from_numpy(np.asarray(triangles))



def to_differential(L,v):
    return L @ v

def from_differential(L, u):
    key = (id(L), 'Cholesky')
    if key not in _cache.keys():
        solver = CholeskySolver(L)
        cache_put(key,solver,L)
    else:
        solver = CholeskySolver(L)
    return solve(solver,u)


def system_matrix(verts,faces,lambda_):
    L = uniform_laplacian_smoothing(verts,faces)
    indexes = torch.arange(verts.shape[0],dtype=torch.int32,device='cuda')
    view = torch.sparse_coo_tensor(torch.stack((indexes,indexes)),torch.ones(verts.shape[0],dtype=torch.float32,device='cuda'), (verts.shape[0],verts.shape[0]))
    return torch.add(view,lambda_*L).coalesce()


def face_normals(verts,faces):
    sort = torch.transpose(faces, 0,1).long().to('cuda')
    verts = torch.transpose(verts,0,1)
    sorted_verts = [verts.index_select(1, sort[0]),
                    verts.index_select(1, sort[1]),
                    verts.index_select(1, sort[2])]
    normals = torch.cross(sorted_verts[1] - sorted_verts[0] , sorted_verts[2] - sorted_verts[0])
    normals = normals / torch.norm(normals,dim=0)
    return normals


def vertex_normals(verts,faces,face_normals):
    sort = torch.transpose(faces, 0,1).long().to('cuda')
    verts = torch.transpose(verts,0,1).to('cuda')
    sorted_verts = [verts.index_select(1, sort[0]),
                    verts.index_select(1, sort[1]),
                    verts.index_select(1, sort[2])]

    vert_normals = torch.zeros_like(verts)

    for i in range(0,3):
        nbs = [(i +1) % 3 , (i +2) % 3]
        n1 = sorted_verts[nbs[0]] - sorted_verts[i]
        n1 = n1 / torch.norm(n1)
        n2 = sorted_verts[nbs[1]] - sorted_verts[i]
        n2 = n2 / torch.norm(n2)
        angle = torch.acos(torch.sum(n1*n2,0).clamp(-1,1))
        normals = face_normals * angle
        for j in range(0,3):
            vert_normals[j].index_add_(0,sort[i],normals[j])
    return (vert_normals / torch.norm(vert_normals,dim=0)).transpose(0,1)


def uniform_laplacian_smoothing(verts, faces):

    i_neighbor = faces[:, [1,2,0]].flatten()
    j_neighbor = faces[:, [2,0,1]].flatten()
    adjacent_verts = torch.stack([torch.cat([i_neighbor,j_neighbor]),torch.cat([j_neighbor,i_neighbor])],dim=0).unique(dim=1)
    adjacent_coords = torch.ones(adjacent_verts.shape[1],device='cuda',dtype=torch.float)
    diagonal_idx = adjacent_verts[0]
    sparse_matrix = torch.cat((adjacent_verts,torch.stack((diagonal_idx,diagonal_idx),dim=0)),dim=1)
    adjacent_coords = torch.cat((-adjacent_coords,adjacent_coords))
    return torch.sparse_coo_tensor(sparse_matrix,adjacent_coords, (verts.shape[0],verts.shape[0])).coalesce()



def to_numpy(tensor_):
    return tensor_.cpu().numpy()

def save_data(verts,faces,filepath):

    print(f"Saving data to ../outputs/{filepath}.ply")
    path = os.path.join(os.getcwd(),"outputs")
    filepath += ".ply" 
    write_ply(os.path.join(path,filepath), verts,faces,ascii=True)

def write_leftovers(faces,filepath):
    path = os.path.join(os.getcwd(),"outputs")
    filepath += ".ply" 
    with open(os.path.join(path,filepath),"a") as fp:
        for f in faces:
            fp.write(f"3 {f[0]} {f[1]}  {f[2]}\n")
        fp.close()

def read_cfg():
    formatting = ["save_path","iterations","step_size","lambda"]
    data = []
    cfg_path = os.path.join(os.getcwd(), "config.txt")
    with open(cfg_path,"r") as cfg:
        lines = cfg.readlines()
        count = 0
        for l in lines:
            l_str = l.split("=")
            if l_str[0] == formatting[count]:
               data.append(l_str[1][:-1])                
            count +=1
        cfg.close()
    return data
