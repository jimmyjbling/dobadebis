import numpy as np
import yaml
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import KDTree
import itertools
import mcubes
from skimage.measure import label

#################################################################
###                Globals                                    ###
#################################################################

atomic_parameters = {}
general_parameters = {}
scalar_factor = -1
trans_func = None
box_dim = None

#################################################################
###                Read Parameters & data                     ###
#################################################################


def read_atomic_properties(yaml_file):
    """

    :param yaml_file:
    :return:
    """
    global atomic_parameters
    file = open(yaml_file, "r")
    atomic_parameters = yaml.load(file, yaml.Loader)
    file.close()


def read_general_properties(yaml_file):
    """

    :param yaml_file:
    :return:
    """
    global general_parameters
    file = open(yaml_file, "r")
    general_parameters = yaml.load(file, yaml.Loader)
    file.close()


def parse_pdb(filename):
    """
    Parses the pdb file to extract the polymer atoms and the inorganic atoms. Will also classify which inorganic class is the ligand
    based on how many inorganic atmos there are, assuming it will have the most. NOTE in order to aviod issues misclassifying ligand,
    it should be defined in the config file. EX: "ligand: AZM"
    :param filename: pdb file to parse
    :return: atoms of ligand, inorganic and polymer
    """
    with open(filename, "r") as file:
        atoms = []
        ligand_atoms = []
        inorganic_atoms = []
        het_atom_types = []
        ligand_chain = None  # NOTE this is so that if multiple copies of same ligand are present, only 1 is selected

        if general_parameters is not None:
            if 'ligand' in general_parameters:
                het_atom_types = [(general_parameters['ligand'], 999999)]

        for line in file:
            split_line = [x for x in line.split(" ") if x != ""]
            if split_line[0] == "HET":
                if split_line[4] == "\n":
                    het_atom_types.append((split_line[1], int(split_line[3])))
                else:
                    het_atom_types.append((split_line[1], int(split_line[4])))
                # print(het_atom_types)
                het_atom_types.sort(key=lambda x: x[1], reverse=True)
            if line[:4] == "ATOM":
                # see page 188 of pdb format version 4 for indexing assignments
                record_name = line[:6]
                atom_id = line[6:11]
                atom_name = line[12:16]
                alt_loc = line[16]
                residue_name = line[17:20]
                chain_id = line[21]
                residue_id = line[22:26]
                i_code = line[26]
                x_coord = line[30:38]
                y_coord = line[38:46]
                z_coord = line[46:54]
                occupancy = line[54:60]
                temp_factor = line[60:66]
                element = line[76:78]
                element = "".join([x for x in element if x != " "])  # removes possible extra space in element name
                charge = line[78:80]
                coords = [float(x_coord), float(y_coord), float(z_coord)]

                atoms.append(tuple([coords, atom_name, element]))
                # atoms.append(tuple([coords, parameters["vdw"][element], atom_name, element]))
            if line[:6] == "HETATM":
                record_name = line[:6]
                atom_id = line[6:11]
                atom_name = line[12:16]
                alt_loc = line[16]
                residue_name = line[17:20]
                chain_id = line[21]
                residue_id = line[22:26]
                i_code = line[26]
                x_coord = line[30:38]
                y_coord = line[38:46]
                z_coord = line[46:54]
                occupancy = line[54:60]
                temp_factor = line[60:66]
                element = line[76:78]
                element = "".join([x for x in element if x != " "])  # removes possible extra space in element name
                charge = line[78:80]
                coords = [float(x_coord), float(y_coord), float(z_coord)]
                # print(residue_name)
                if residue_name.replace(" ", "") == het_atom_types[0][0]:
                    if ligand_chain is None:
                        ligand_atoms.append(tuple([coords, atom_name, element]))
                        ligand_chain = chain_id
                    else:
                        if chain_id == ligand_chain:
                            ligand_atoms.append(tuple([coords, atom_name, element]))
                        else:
                            inorganic_atoms.append(tuple([coords, atom_name, element]))
                else:
                    inorganic_atoms.append(tuple([coords, atom_name, element]))
        file.close()
    return atoms, ligand_atoms, inorganic_atoms


def reduce_atoms(atoms, ligand_atoms):
    ligand_tree = KDTree([x[0] for x in atoms])
    total_neighbors = []
    for latom in ligand_atoms:
        coords = latom[0]
        neighbors = ligand_tree.query_ball_point(coords, 20.0)
        total_neighbors = total_neighbors + neighbors
    total_neighbors = list(set(total_neighbors))
    neighbor_atoms = [atoms[x] for x in total_neighbors]
    return neighbor_atoms


#################################################################
###            Build and fill transformed grid                ###
#################################################################


def transform_scale(atoms, typ=1):
    """

    :param atoms:
    :param typ:
    :return:
    """

    def get_coords(atoms):
        """

        :param atoms:
        :return:
        """
        coords = []
        for atom in atoms:
            coords.append(atom[0])
        return coords

    global scalar_factor
    global trans_func
    global box_dim

    # read in parameters set by user
    box_length = int(atomic_parameters['resolution_default'])
    probe_radius = float(atomic_parameters['probe_radius'])
    scale_factor_default_value = float(atomic_parameters['scale_factor_default_value'])
    scale_factor_default = bool(atomic_parameters['scale_factor_default'])
    max_box_length = int(atomic_parameters['max_box_length'])

    # constant to correct for lack of radii in finding min max points (generous estimate)
    buffer = 2.5

    # get all the coordinate data of the atoms
    coords = get_coords(atoms)

    # for van der waal surface generation only
    if typ == 0:
        x_min = min([coord[0] for coord in coords]) - buffer
        x_max = max([coord[0] for coord in coords]) + buffer
        y_min = min([coord[1] for coord in coords]) - buffer
        y_max = max([coord[1] for coord in coords]) + buffer
        z_min = min([coord[2] for coord in coords]) - buffer
        z_max = max([coord[2] for coord in coords]) + buffer
    # for SA surface and molecular (SE) surface only
    elif typ == 1:
        x_min = min([coord[0] for coord in coords]) - (buffer + probe_radius)
        x_max = max([coord[0] for coord in coords]) + (buffer + probe_radius)
        y_min = min([coord[1] for coord in coords]) - (buffer + probe_radius)
        y_max = max([coord[1] for coord in coords]) + (buffer + probe_radius)
        z_min = min([coord[2] for coord in coords]) - (buffer + probe_radius)
        z_max = max([coord[2] for coord in coords]) + (buffer + probe_radius)
    # return error when type of surface not correctly defined
    else:
        return -1

    # set transformation values of each point to center minimum point at origin (0,0,0)
    transform_x = 0 - x_min
    transform_y = 0 - y_min
    transform_z = 0 - z_min

    # determine which axes is the longest
    max_length = max([x_max - x_min, y_max - y_min, z_max - z_min])

    # scale points up to fill entire axes of the largest axes based on box length
    scale_factor = (box_length - 1) / max_length

    if scale_factor_default:
        box_length = (scale_factor_default_value / scale_factor) * box_length
        scale_factor = scale_factor_default_value
        if box_length > max_box_length:
            scale_factor = float(scale_factor * (max_box_length / box_length))
            box_length = max_box_length

    x_axis = np.ceil((x_max - x_min) * scale_factor)
    y_axis = np.ceil((y_max - y_min) * scale_factor)
    z_axis = np.ceil((z_max - z_min) * scale_factor)

    box_dimensions = [x_axis, y_axis, z_axis]
    transform_func = [transform_x, transform_y, transform_z]

    # set global transform variables
    scalar_factor = scale_factor
    trans_func = transform_func
    box_dim = box_dimensions

    return transform_func, box_dimensions, scale_factor


def fill_voxels_old(atoms, surface_type=1):
    """

    :param atoms:
    :param surface_type: type of surface. 0 for vdw, 1 for SASA, 2 for molecular
    :return:
    """

    probe_radius = atomic_parameters["probe_radius"]

    binary_image = np.full((int(box_dim[0]), int(box_dim[1]), int(box_dim[2])), 0, dtype=int)

    # color dictionary for each voxel. Holds which atoms are close, how close and the atoms color for each voxel
    voxel_atom_proximity = {}

    vdw_type = str(atomic_parameters['vdw_type'])
    vdw_dict = None

    if vdw_type == "basic":
        vdw_dict = atomic_parameters['vdw_basic']
    else:
        print("Van Der Waal radii type parameters missing/incorrect. Check atomic_config file")

    if vdw_dict is None:
        print("Van Der Waal radii set nonexistent. Check atomic_config file")

    for atom in atoms:
        scaled_vdw = 1.7 * scalar_factor  # defaults to carbon if atom not found
        try:
            if surface_type == 0:
                scaled_vdw = vdw_dict[atom[2]] * scalar_factor
            else:
                scaled_vdw = (vdw_dict[atom[2]] + probe_radius) * scalar_factor
        except KeyError:
            print("atoms type", atom[2], "not found. Defaulting to C atom vdw of 1.7")

        atom_coords = atom[0]

        trans_cp_x = (atom_coords[0] + trans_func[0]) * scalar_factor
        trans_cp_y = (atom_coords[1] + trans_func[1]) * scalar_factor
        trans_cp_z = (atom_coords[2] + trans_func[2]) * scalar_factor

        center_point = [trans_cp_x, trans_cp_y, trans_cp_z]

        x_face_pos = int(np.floor(center_point[0] + scaled_vdw))
        x_face_neg = int(np.ceil(center_point[0] - scaled_vdw))
        y_face_pos = int(np.floor(center_point[1] + scaled_vdw))
        y_face_neg = int(np.ceil(center_point[1] - scaled_vdw))
        z_face_pos = int(np.floor(center_point[2] + scaled_vdw))
        z_face_neg = int(np.ceil(center_point[2] - scaled_vdw))

        possible_voxels = list(itertools.product(range(x_face_neg, x_face_pos + 1),
                                                 range(y_face_neg, y_face_pos + 1),
                                                 range(z_face_neg, z_face_pos + 1)))
        # print(time.time() - start)
        check = time.time()
        for pos_voxel in possible_voxels:
            dist_from_voxel_center = (pos_voxel[0] - center_point[0]) ** 2 + (pos_voxel[1] - center_point[1]) ** 2 + \
                                     (pos_voxel[2] - center_point[2]) ** 2
            if dist_from_voxel_center < scaled_vdw ** 2:
                binary_image[pos_voxel[0], pos_voxel[1], pos_voxel[2]] = 1
    return binary_image


######################################
###            color               ###
######################################


def map_color_to_surface_rough(atoms, verts):
    tree = KDTree([scalar_factor * (np.array(x[0]) + trans_func) for x in atoms])
    colors = []

    for vert in verts:
        nearest = tree.query(vert)
        nearest_atom = atoms[nearest[1]]
        color = atomic_parameters["color"][nearest_atom[2]].split(",")
        colors.append(color)
    return colors


def map_colors_to_molecular_surf_smooth(atoms, verts):
    def threed_dist(coord1, coord2):
        x1 = coord1[0]
        y1 = coord1[1]
        z1 = coord1[2]
        x2 = coord2[0]
        y2 = coord2[1]
        z2 = coord2[2]

        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

        return d

    tree = KDTree([scalar_factor * (np.array(x[0]) + trans_func) for x in atoms])
    vdw_dict = atomic_parameters['vdw_basic']
    colors = []
    for vert in verts:
        neighbors = tree.query_ball_point(vert, 4.0 * scalar_factor)
        # Temp fix to color map issue, assume carbon color if out of range
        if not neighbors:
            colors.append([0, 1, 0])
            continue
        min_distance_from_vdw = []
        for neighbor in neighbors:
            atom_name = atoms[neighbor][2]
            scaled_vdw = vdw_dict[atom_name] * scalar_factor
            distance = threed_dist(scalar_factor * (np.array(atoms[neighbor][0]) + trans_func), vert)
            error_dist = np.abs(distance-scaled_vdw)
            if error_dist < 4.0 * scalar_factor:
                min_distance_from_vdw.append((error_dist, atoms[neighbor]))
        colors_a = []
        errors = []
        for x in min_distance_from_vdw:
            colors_a.append(atomic_parameters["color"][x[1][2]].split(","))
            errors.append(x[0]**-2)
        total_dist = sum(errors)
        total_red = 0.0
        total_blue = 0.0
        total_green = 0.0

        for color, distance in list(zip(colors_a, errors)):
            color = [int(x) for x in color]
            total_red += (color[0] * distance)
            total_blue += (color[1] * distance)
            total_green += (color[2] * distance)

        new_red = int(round(total_red / total_dist))
        new_blue = int(round(total_blue / total_dist))
        new_green = int(round(total_green / total_dist))

        new_color = tuple([new_red, new_blue, new_green])
        colors.append(new_color)
    return colors


####################################################
###               Build surfaces                 ###
####################################################


def molecular_surface(binary_image):
    probe_radius = float(atomic_parameters['probe_radius'])
    probe_radius_s = probe_radius * scalar_factor
    edt = distance_transform_edt(binary_image)
    mol_sur = np.where(edt >= probe_radius_s, 1, 0)
    return mol_sur


def circle_roll(mol_surf, radius):
    radius = radius * scalar_factor
    # invert the image so processing the pocket can occur
    mol_surf = np.where(mol_surf == 1, 0, 1)
    # do a EDT distance transform
    mol_surf = distance_transform_edt(mol_surf)
    # set all the points where the distance was larger than the radius as 1. These are all the points that if a ball
    # with the given radius was centered on, the ball would not impact the protein
    mol_surf = np.where(mol_surf >= radius, 1, 0)
    # "uninvert" the image, as it now lost the bad points that were unreachable by the probe
    mol_surf = np.where(mol_surf == 1, 0, 1)
    # do a second edt to re add on the radius that was taken off during the first edt, but now only on good points
    mol_surf = distance_transform_edt(mol_surf)
    # select the points larger than the radius to get the probe rolled surface
    mol_surf = np.where(mol_surf >= radius, 1, 0)

    return mol_surf


def circle_roll_invert_at_end(mol_surf, radius, ligand):
    radius = radius * scalar_factor
    # do a EDT distance transform
    mol_surf = distance_transform_edt(mol_surf)
    # set all the points where the distance was larger than the radius as 1. These are all the points that if a ball
    # with the given radius was centered on, the ball would not impact the protein
    mol_surf = np.where(mol_surf >= radius, 1, 0)

    # select the group that is only by the ligand of interest
    labeled_mol_surf = label(mol_surf)

    labels = {}
    for atom in ligand:
        coord = atom[0]
        trans_lig_x = int((coord[0] + trans_func[0]) * scalar_factor)
        trans_lig_y = int((coord[1] + trans_func[1]) * scalar_factor)
        trans_lig_z = int((coord[2] + trans_func[2]) * scalar_factor)
        lig_label = labeled_mol_surf[trans_lig_x][trans_lig_y][trans_lig_z]
        if lig_label in labels.keys():
            labels[lig_label] = labels[lig_label] + 1
        else:
            labels[lig_label] = 1

    ligand_label = int(max(labels, key=labels.get))

    #ligand_label = labeled_mol_surf[lcx][lcy][lcz]
    mol_surf = np.where(labeled_mol_surf == ligand_label, 1, 0)

    # "uninvert" the image, as it now lost the bad points that were unreachable by the probe
    mol_surf = np.where(mol_surf == 1, 0, 1)

    # do a second edt to re add on the radius that was taken off during the first edt, but now only on good points
    mol_surf = distance_transform_edt(mol_surf)
    # select the points larger than the radius to get the probe rolled surface
    mol_surf = np.where(mol_surf >= radius, 1, 0)
    # invert the image so processing the pocket can occur
    mol_surf = np.where(mol_surf == 1, 0, 1)
    return mol_surf


def get_pocket_space(big_probe, small_probe):
    merged_surf = np.where(np.logical_and(big_probe == 1, small_probe != 1), 1, 0)
    return merged_surf


####################################################
###               Clean surfaces                 ###
####################################################


def remove_branching():
    return None

def ligand_center_point(ligand):
    coords = np.array([x[0] for x in ligand])
    means = (coords.mean(axis=0) + np.array(trans_func)) * scalar_factor
    return int(round(means[0])), int(round(means[1])), int(round(means[2]))


def reduce_pocket_space(pocket_space, ligand):

    reduced_space = circle_roll_invert_at_end(pocket_space, 1.5, ligand)

    """
    pocket_voxels = np.where(pocket_space == 1)
    reduced_pocket_space = np.full((int(box_dim[0]), int(box_dim[1]), int(box_dim[2])), 0, dtype=int)
    for i in range(len(pocket_voxels[0])):
        x, y, z = (pocket_voxels[0][i], pocket_voxels[1][i], pocket_voxels[2][i])
        surrounding_points = []
        for a in [-1, 0, 1]:
            for b in [-1, 0, 1]:
                for c in [-1, 0, 1]:
                    surrounding_points.append([x+a, y+b, z+c])
        surrounding_points.remove([x, y, z])
        surrounding_points = [d for d in surrounding_points if d[0] >= 0 and d[1] >= 0 and d[2] >= 0 and
                              d[0] < box_dim[0] and d[1] < box_dim[1] and d[2] < box_dim[2]]
        surround_score = 0
        for surrounding_point in surrounding_points:
            sx = surrounding_point[0]
            sy = surrounding_point[1]
            sz = surrounding_point[2]
            if pocket_space[sx][sy][sz] == 1:
                surround_score = surround_score + 1
        if surround_score >= 15:
            reduced_pocket_space[x][y][z] = 1
    return reduced_pocket_space
    """
    return reduced_space

def un_tranform_verts(verts):
    un_trans_verts = verts.copy()
    for i in range(len(verts)):
        vert = verts[i]
        x = (vert[0] / scalar_factor) - trans_func[0]
        y = (vert[1] / scalar_factor) - trans_func[1]
        z = (vert[2] / scalar_factor) - trans_func[2]
        un_trans_verts[i] = [x, y, z]
    return un_trans_verts


def final_labeling(ideal):
    labeled = label(ideal)
    end = np.where(labeled == 3, 1, 0)
    return end


################################################
###           Build Files                    ###
################################################


def build_ply_file(verts, faces, colors, out_name):
    with open(out_name, "w") as f:
        f.write("ply\n"
                "format ascii 1.0\n"
                "element vertex " + str(len(verts)) + "\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "element face " + str(len(faces)) + "\n"
                "property list uchar int vertex_indices\n"
                "end_header\n")
        for vert, col in zip (verts, colors):
            write_str = str(round(vert[0], 5)) + " " + str(round(vert[1], 5)) + " " + str(round(vert[2], 5)) + " " + \
                        str(col[0]) + " " + str(col[1]) + " " + str(col[2]) + " \n"
            f.write(write_str)
        for face in faces:
            write_str = "3 " + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + " \n"
            f.write(write_str)
        f.close()


def build_isolated_vrml_file(nvert, ntri, ncolor, out_name):
    with open(out_name, "w") as f:
        f.write("#VRML V2.0 utf8\n")
        f.write("NavigationInfo {\n")
        f.write('type [ "EXAMINE", "ANY" ]\n')
        f.write('}\n')
        f.write('Transform {\n')
        f.write('scale 1 1 1\n')
        f.write('translation 0 0 0\n')
        f.write('children\n')
        f.write('[\n')
        f.write('Shape\n')
        f.write('{\n')
        f.write('geometry IndexedFaceSet\n')
        f.write('{\n')
        f.write('creaseAngle .5\n')
        f.write('solid FALSE\n')
        f.write('coord Coordinate\n')
        f.write('{\n')
        f.write('point\n')
        f.write('[\n')

        for vert in nvert:
            write_str = str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + ", \n"
            f.write(write_str)

        f.write(']\n')
        f.write('}\n')
        if ncolor is not None:
            f.write("color Color\n")
            f.write("{\n")
            f.write("color\n")
            f.write("[\n")
            for color in ncolor:
                write_str = str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + ", \n"
                f.write(write_str)
            f.write(']\n')
            f.write('}\n')
        f.write("coordIndex\n")
        f.write("[\n")
        for coord in ntri:
            write_str = str(coord[0]) + "," + str(coord[1]) + "," + str(coord[2]) + ",-1, \n"
            f.write(write_str)
        f.write(']\n')
        f.write('}\n')
        f.write("appearance Appearance\n")
        f.write("{\n")
        f.write("material Material\n")
        f.write("{\n")
        f.write('ambientIntensity 0.2\ndiffuseColor 0.9 0.9 0.9\nspecularColor .1 .1 .1\nshininess .5\n')
        f.write("}\n")
        f.write("}\n")
        f.write("}\n")
        f.write("]\n")
        f.write("}\n")
    f.close()


####################################
###          MAIN                ###
####################################
with open("DUD-E.data", "r") as f:
    for line in f:
        line = line[:-1]
        # line = "3bkl"  # debug line
        ideal_out_name = str(line) + "_ideal_rough_color.ply"
        real_out_name = str(line) + "_actual_ligand.ply"
        protein_out_name = str(line) + "_binding_site.ply"
        pdb_file = "DUD-E_PDB_Files/" + str(line) + ".pdb"

        # read parameters
        start = time.time()
        read_atomic_properties("atomic_config.yaml")
        read_general_properties("general_config.yaml")


        # load pdb file
        check0 = time.time()
        atoms, ligand, inorganic = parse_pdb(pdb_file)

        # reduce the area of interest to just that around the ligand
        check1 = time.time()
        reduced = reduce_atoms(atoms, ligand)

        # transform the space to larger box for higher resolution
        check2 = time.time()
        trans, box, scale = transform_scale(reduced)
        lcx, lcy, lcz = ligand_center_point(ligand)

        # fill in the impacted protein voxels TODO improve this
        check3 = time.time()
        binary_image = fill_voxels_old(reduced)

        # calculate the molecular surface
        check4 = time.time()
        binary_image_mol = molecular_surface(binary_image)

        # roll a big and small probe ball across the surface
        check5 = time.time()
        small_probe = circle_roll(binary_image_mol, 1.1)
        big_probe = circle_roll(binary_image_mol, 6)

        # define the pocket space as the area between the small probe and large probe surface
        check6 = time.time()
        pocket_space = get_pocket_space(big_probe, small_probe)

        # clean up the pocket space
        check7 = time.time()
        ideal_space = reduce_pocket_space(pocket_space, ligand)

        # smooth image for cleaner image
        check8 = time.time()
        smoothed = mcubes.smooth(ideal_space)

        # MMC to turn image into surface
        check9 = time.time()
        vertices_s, triangles_s = mcubes.marching_cubes(smoothed, 0)
        vertices_trans = un_tranform_verts(vertices_s)

        # map colors to the new surface based on near by atom colors TODO improve this
        check10 = time.time()
        color = map_color_to_surface_rough(atoms, vertices_s)

        # build VRML file of mesh surface
        check11 = time.time()
        build_ply_file(vertices_trans, triangles_s, color, out_name=ideal_out_name)

        ######################################
        ###   Build Real Ligand Surface    ###
        ######################################

        check12 = time.time()
        ligand_binary_image = fill_voxels_old(ligand)
        mol_ligand_binary_image = molecular_surface(ligand_binary_image)
        ligand_smoothed = mcubes.smooth(mol_ligand_binary_image)
        vertices_lig, triangles_lig = mcubes.marching_cubes(ligand_smoothed, 0)
        vertices_trans_lig = un_tranform_verts(vertices_lig)
        color_lig = map_colors_to_molecular_surf_smooth(ligand, vertices_lig)
        build_ply_file(vertices_trans_lig, triangles_lig, color_lig, out_name=real_out_name)
        check13 = time.time()

        #######################################
        ###   Build the binding region      ###
        #######################################

        """
        mol_smooth = mcubes.smooth(binary_image_mol)
        vertices_mol, triangles_mol = mcubes.marching_cubes(mol_smooth, 0)
        vertices_trans_mol = un_tranform_verts(vertices_mol)
        color_mol = map_colors_to_molecular_surf_smooth(atoms, vertices_mol)
        build_ply_file(vertices_trans_mol, triangles_mol, color_mol, out_name=protein_out_name)
        """

        print("Read config:", check0 - start)
        print("Read config:", check0 - start)
        print("Read PDB:", check1 - check0)
        print("Reduce atoms:", check2 - check1)
        print("Scale for greater resolution:", check3 - check2)
        print("Build grid:", check4 - check3)
        print("Determine molecular surface:", check5 - check4)
        print("Rolling balls:", check6 - check5)
        print("Finding pocket:", check7 - check6)
        print("Cleaning pocket:", check8 - check7)
        print("Smoothing:", check9 - check8)
        print("Marching cube:", check10 - check9)
        print("Mapping color:", check11 - check10)
        print("Writing ply file:",  check12 - check11)
        print("Building real ligand surface:", check13 - check12)
        print("Total time:", time.time() - start, "\n")

        # break  # debug line
