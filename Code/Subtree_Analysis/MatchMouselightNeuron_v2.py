import os
from util.swc_branching_points import SWCBranchingPoint, TreeNode


def output_best_swc(best_swc_dict, output_path):
    keys = best_swc_dict.keys()
    keys = sorted(keys)
    with open(output_path, 'w') as file:
        for key in keys:
            node = best_swc_dict[key]
            file.write(f"{node.id} 1 {node.x} {node.y} {node.z} 1 {node.father_id}\n")

if __name__ == '__main__':
    input_mouselight_folder = "data/MouseLight/target_neurons"
    input_DM_swc = "data/MouseLight/Jai_atlas_32_branch40.swc"
    output_matched_subtree_folder = "data/MouseLight/output_matched_subtrees_v2_40_branches"
    branch_cnt_mask = (1<<40) - 1
    if not os.path.exists(output_matched_subtree_folder):
        os.mkdir(output_matched_subtree_folder)

    input_mouselights = sorted([os.path.join(input_mouselight_folder, filename) for filename in os.listdir(input_mouselight_folder) if filename.endswith('.swc')])

    DM_branch_swc = SWCBranchingPoint(input_DM_swc)
    mouselights = [SWCBranchingPoint(path) for path in input_mouselights]
    selected_branch_mask = 0
    for mid, m in enumerate(mouselights):
        print("processing " + input_mouselights[mid])
        mouselight_all_coords = m.get_all_coordinates()
        best_match_mask, best_distance, best_dict = DM_branch_swc.get_matched_subtree_v2(mouselight_all_coords)
        print("best matched with subtree mask:", best_match_mask, "with distance:", best_distance)
        input_name = os.path.splitext(os.path.basename(input_mouselights[mid]))[0]
        input_name = input_name[:input_name.find('_')]
        output_name = f"{input_name}_subtree{best_match_mask}.swc"
        # print(input_name, output_name)
        output_best_swc(best_swc_dict=best_dict, output_path=os.path.join(output_matched_subtree_folder, output_name))
        selected_branch_mask = selected_branch_mask | best_match_mask
    # print(selected_branch_mask)
    unmatched_swc_dict = DM_branch_swc.get_unmatched_subtree(selected_branch_mask)
    unmatched_mask = branch_cnt_mask - selected_branch_mask
    output_best_swc(unmatched_swc_dict, os.path.join(output_matched_subtree_folder, f'unmatched_subtree_{unmatched_mask}.swc'))