# ABAQUS Prefactory Information
def equation_constraint(model, assembly, parent_part_name, child_part_name, nodes_to_link, linked_dof=[1, 2, 3, 4, 5, 6]):
    # nodes_to_link are sets of node labels that correspond to the parent_part_name and child_part_name parts
    for pair in nodes_to_link:
        label_one, label_two = pair[0], pair[1]
        # Create a set for each of these nodes
        assembly.Set(
            name='equation-set-{}-{}-{}-{}-1'.format(parent_part_name, child_part_name, label_one, label_two),
            nodes=assembly.instances[parent_part_name].nodes.sequenceFromLabels((label_one,))
        )
        assembly.Set(
            name='equation-set-{}-{}-{}-{}-2'.format(parent_part_name, child_part_name, label_one, label_two),
            nodes=assembly.instances[child_part_name].nodes.sequenceFromLabels((label_two,))
        )

        for dof in linked_dof:
            model.Equation(
                name='Equation-{}-{}-{}-{}-{}'.format(parent_part_name, child_part_name, label_one, label_two, dof),
                terms=(
                    (-1.0, 'equation-set-{}-{}-{}-{}-1'.format(parent_part_name, child_part_name, label_one, label_two), dof),
                    ( 1.0, 'equation-set-{}-{}-{}-{}-2'.format(parent_part_name, child_part_name, label_one, label_two), dof),
                ),
            )

def equation_sets(model, name, set_one, set_two, linked_dof=[1, 2, 3, 4, 5, 6]):
    """
    Link two sets using equations for each of the requested degrees of freedom
    
    Parameters
    ----------
    model : Abaqus model
    name : str
        Name of the equation as it will appear in the Abaqus tree
    set_one : name of the set of follower nodes
    set_two : name of the set of the main nodes
        Must contain a single 'driving' node for the rest of the set
    linked_dof : list[] containing the requested degrees of freedom to be linked
        x = 1, y = 2, z = 3, rev_x = 4, rev_y = 5, rev_z = 6
    
    Returns
    -------
    
    """

    for dof in linked_dof:
        model.Equation(
            name = '{}-{}'.format(name, dof),
            terms = (
                (1.0, set_one, dof),
                (-1.0, set_two, dof)
            )
        )

    print("[equation_sets] Linked '{}' with {}.".format(set_one, set_two))