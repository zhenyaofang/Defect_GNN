import os
import numpy as np
from tqdm import tqdm
import multiprocessing
import warnings

from ripser import Rips, ripser

small = 0.0001
rs = 0.25
dt = np.dtype([('typ', 'S2'), ('pos', float, (3,))])

def get_prim_structure_info(data_dir, id):
    with open(data_dir + '/structures/' + id + '.vasp', 'r') as f:
        lines = f.read().splitlines()
    atom_map = []
    index_up = 0
    for atom, num in zip(lines[5].split(), lines[6].split()):
        index_up += int(num)
        atom_map.append((atom, index_up))
    lattice_vec = []
    for line in lines[2:5]:
        x, y, z = line.split()
        lattice_vec.append([float(x), float(y), float(z)])
    lattice_vec = np.array(lattice_vec)
    # get atom position
    index_atom = 0
    atom_nums = len(lines[8:])
    atom_vec = np.zeros([atom_nums], dtype=dt)
    for i in range(atom_nums):
        line = lines[8 + i]
        x, y, z, type = line.split()
        if i < atom_map[index_atom][1]:
            atom_vec[i]['typ'] = atom_map[index_atom][0]
        else:
            index_atom += 1
            atom_vec[i]['typ'] = atom_map[index_atom][0]
        atom_vec[i]['pos'][:] = np.array([float(x), float(y), float(z)])

    if not os.path.exists(data_dir + "/atoms"):
        os.makedirs(data_dir + "/atoms")

    with open(data_dir + '/atoms/' + id + '_original.npz', 'wb') as out_file:
        np.savez(out_file, lattice_vec=lattice_vec, atom_vec=atom_vec)

def enlarge_cell(data_dir, id, r_cutoff):
    with open(data_dir + '/atoms/' + id + '_original.npz', 'rb') as structfile:
        data = np.load(structfile)
        lattice_vec = data['lattice_vec'];
        atom_vec = data['atom_vec']
    min_lattice = min([np.linalg.norm(i) for i in lattice_vec])
    mul_time = int(np.ceil(r_cutoff / min_lattice))
    center_atom_vec = atom_vec.copy()
    center_atom_vec['pos'][:] += mul_time
    center_atom_vec['pos'][:] = np.matmul(center_atom_vec['pos'][:], lattice_vec)

    enlarge_dict = {}
    atom_nums = 0
    for atom in atom_vec:
        typ = atom['typ']
        tmp = []
        if typ not in enlarge_dict:
            enlarge_dict[typ] = []
        for i in range(mul_time * 2 + 2):
            tmp.append(atom['pos'][0] + i)
            for j in range(mul_time * 2 + 2):
                tmp.append(atom['pos'][1] + j)
                for k in range(mul_time * 2 + 2):
                    tmp.append(atom['pos'][2] + k)
                    point = tmp.copy()
                    if point not in enlarge_dict[typ]:
                        enlarge_dict[typ].append(point)
                        atom_nums += 1
                    tmp.pop()
                tmp.pop()
            tmp.pop()
    # print(enlarge_dict, atom_nums)
    enlarge_vec = np.zeros([atom_nums], dtype=dt)
    cart_enlarge_vec = np.zeros([atom_nums], dtype=dt)
    atom_index = 0
    for typ, vec in enlarge_dict.items():
        for v in vec:
            enlarge_vec[atom_index]['typ'] = typ
            cart_enlarge_vec[atom_index]['typ'] = typ
            enlarge_vec[atom_index]['pos'][:] = np.array(v)
            cart_enlarge_vec[atom_index]['pos'][:] = np.matmul(np.array(v), lattice_vec)
            atom_index += 1
    with open(data_dir + '/atoms/' + id + '_' + str(r_cutoff) + '_enlarge.npz', 'wb') as out_file:
        np.savez(out_file, CAV=center_atom_vec, CEV=cart_enlarge_vec)

def get_betti_num(data_dir, id, r_cutoff):
    if os.path.exists(data_dir + '/betti_num/' + id + '_' + str(r_cutoff)):
        return "exists"
    with open(data_dir + '/atoms/' + id + '_' + str(r_cutoff) + '_enlarge.npz', 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec = data['CAV'];
        cart_enlarge_vec = data['CEV']
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1

    if not os.path.exists(data_dir + "/betti_num"):
        os.makedirs(data_dir + "/betti_num")

    out_File = open(data_dir + '/betti_num/' + id + '_' + str(r_cutoff), 'w')
    for cav in center_atom_vec:
        center_atom_type = cav['typ'].decode()
        for ele in typ_dict.keys():
            # get atom position in each pair
            pair_index = []
            for i in range(len(cart_enlarge_vec)):
                vec = cart_enlarge_vec[i]
                # make change
                # if (vec['typ'].decode() == ele or vec['typ'].decode() == center_atom_type) and np.linalg.norm(
                #         vec['pos'][:] - cav['pos'][:]) <= r_cutoff:
                if vec['typ'].decode() == ele and np.linalg.norm(vec['pos'][:] - cav['pos'][:]) <= r_cutoff:
                    pair_index.append(i)
            if len(pair_index) == 0:
                continue
            points_num = len(pair_index)
            pair_pos = np.zeros((points_num + 1, 3))
            pair_pos[0][:] = cav['pos'][:]
            index = 1
            for i in pair_index:
                atom = cart_enlarge_vec[i]['pos']
                pair_pos[index][:] = np.array([atom[0], atom[1], atom[2]])
                index += 1

            # calculate barcode
            dgms = ripser(pair_pos, maxdim=2, thresh=r_cutoff)['dgms']
            for i, dgm in enumerate(dgms):
                for p in dgm:
                    out_File.write(center_atom_type + ele + ' ' + str(i) + ' ' + str(p[0]) + ' ' + str(p[1]) + '\n')
        out_File.write('\n')
    out_File.close()

def getElementalBettiProperties(lines, typ_dict):
    Bar0Death = []; Bar1Birth = []; Bar1Death = []; Bar2Birth = []; Bar2Death = []
    WBar0Death = []; WBar1Birth = []; WBar1Death = []; WBar2Birth = []; WBar2Death = []

    for line in lines:
        typ, dim, birth, death = line.split()
        center_atom = typ[0:2] if typ[1].islower() else typ[0]
        ca_num = typ_dict[center_atom]
        dim = int(dim); birth = float(birth); death = float(death)

        # Birth
        if dim == 1:
            Bar1Birth.append(birth)
            WBar1Birth.append(birth / ca_num)
        elif dim == 2:
            Bar2Birth.append(birth)
            WBar2Birth.append(birth / ca_num)
        # Death
        if death == float('inf'): continue
        if dim == 0:
            Bar0Death.append(death)
            WBar0Death.append(death / ca_num)
        elif dim == 1:
            Bar1Death.append(death)
            WBar1Death.append(death / ca_num)
        elif dim == 2:
            Bar2Death.append(death)
            WBar2Death.append(death / ca_num)

    Bar0Death = np.asarray(Bar0Death); Bar1Birth = np.asarray(Bar1Birth); Bar1Death = np.asarray(Bar1Death); Bar2Birth = np.asarray(Bar2Birth); Bar2Death = np.asarray(Bar2Death);
    WBar0Death = np.asarray(WBar0Death); WBar1Birth = np.asarray(WBar1Birth); WBar1Death = np.asarray(WBar1Death); WBar2Birth = np.asarray(WBar2Birth); WBar2Death = np.asarray(WBar2Death);
    Feature_2 = []
    # Betti0
    if len(Bar0Death) > 0:
        Feature_2.append(np.mean(Bar0Death, axis=0))
        Feature_2.append(np.std(Bar0Death, axis=0))
        Feature_2.append(np.max(Bar0Death, axis=0))
        Feature_2.append(np.min(Bar0Death, axis=0))
        Feature_2.append(np.sum(WBar0Death, axis=0))
    else:
        Feature_2.extend([0.] * 5)
    # Betti1
    if len(Bar1Death) > 0:
        Feature_2.append(np.mean(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.std(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.max(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.min(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.sum(WBar1Death - WBar1Birth, axis=0))
        Feature_2.append(np.mean(Bar1Birth, axis=0))
        Feature_2.append(np.std(Bar1Birth, axis=0))
        Feature_2.append(np.max(Bar1Birth, axis=0))
        Feature_2.append(np.min(Bar1Birth, axis=0))
        Feature_2.append(np.sum(WBar1Birth, axis=0))
        Feature_2.append(np.mean(Bar1Death, axis=0))
        Feature_2.append(np.std(Bar1Death, axis=0))
        Feature_2.append(np.max(Bar1Death, axis=0))
        Feature_2.append(np.min(Bar1Death, axis=0))
        Feature_2.append(np.sum(WBar1Death, axis=0))
    else:
        Feature_2.extend([0.] * 15)
    # Betti2
    if len(Bar2Death) > 0:
        Feature_2.append(np.mean(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.std(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.max(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.min(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.sum(WBar2Death - WBar2Birth, axis=0))
        Feature_2.append(np.mean(Bar2Birth, axis=0))
        Feature_2.append(np.std(Bar2Birth, axis=0))
        Feature_2.append(np.max(Bar2Birth, axis=0))
        Feature_2.append(np.min(Bar2Birth, axis=0))
        Feature_2.append(np.sum(WBar2Birth, axis=0))
        Feature_2.append(np.mean(Bar2Death, axis=0))
        Feature_2.append(np.std(Bar2Death, axis=0))
        Feature_2.append(np.max(Bar2Death, axis=0))
        Feature_2.append(np.min(Bar2Death, axis=0))
        Feature_2.append(np.sum(WBar2Death, axis=0))
    else:
        Feature_2.extend([0.] * 15)
    Feature_2 = np.asarray(Feature_2, float)

    return Feature_2

def getElementalBettiFeatures(data_dir, id, r_cutoff=10):
    with open(data_dir + '/atoms/' + id + '_' + str(r_cutoff) + '_enlarge.npz', 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec = data['CAV']
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1

    with open(data_dir + '/betti_num/' + id + '_' + str(r_cutoff), 'r') as phfile:
        lines = phfile.read().splitlines()

    Feature = []
    current_block = []
    for line in lines:
        if line.strip():
            current_block.append(line)
        else:
            Feature.append(getElementalBettiProperties(current_block, typ_dict))
            current_block = []

    if current_block:
        Feature.append(getElementalBettiProperties(current_block, typ_dict))

    Feature = np.asarray(Feature, float)
    return Feature

def calculateBettiNumber(ids, data_dir, r_cutoff, chunk_id):
    for id in tqdm(ids, desc=f'Process {chunk_id}'):
        get_prim_structure_info(data_dir, id)
        enlarge_cell(data_dir, id, r_cutoff)
        get_betti_num(data_dir, id, r_cutoff)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    data_dir = './datasets/raw'
    r_cutoff = 2.5

    files = os.listdir(f'{data_dir}/structures')
    ids = [file.split('.')[0] for file in files]

    num_processes = multiprocessing.cpu_count()
    chunk_size = len(ids) // num_processes
    id_chunks = [ids[i:i+chunk_size] for i in range(0, len(ids), chunk_size)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(calculateBettiNumber, [(chunk, data_dir, r_cutoff, chunk_id) for chunk_id, chunk in enumerate(id_chunks)])

    #files = files[ : len(files) // 4]
    for file in tqdm(files, total=len(files)):
        id = file.split('.')[0]
        calculateBettiNumber(data_dir=data_dir, id=id, r_cutoff=r_cutoff)