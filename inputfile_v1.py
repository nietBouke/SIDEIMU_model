# =============================================================================
# code where I group the physical and geometric input for the delta in a class
# =============================================================================
import sys

class input_network:
    #load
    from Properties_networks.gegs_Kapaus import inp_Kapaus1_geo, inp_Kapaus1_phys
    from Properties_networks.gegs_RedRiver import inp_RedRiver1_geo, inp_RedRiver2_geo, inp_RedRiver1_phys, inp_RedRiver2_phys
    from Properties_networks.gegs_delta_test import inp_deltatest1_geo, inp_deltatest2_geo, inp_deltatest3_geo, inp_deltatest4_geo, inp_deltatest5_geo ,\
        inp_deltatest1_phys, inp_deltatest2_phys, inp_deltatest3_phys,  inp_deltatest4_phys, inp_deltatest5_phys
    from Properties_networks.gegs_Po import inp_Po2_geo, inp_Po2_phys, inp_Po3_phys
    from Properties_networks.gegs_RijnMaas_try import inp_RijnMaas0_geo, inp_RijnMaas0_highres_geo, inp_RijnMaas0_higherres_geo, inp_RijnMaas1_geo, inp_RijnMaas2_geo, inp_RijnMaas3_geo, inp_RijnMaas0_phys, inp_RijnMaas1_phys, inp_RijnMaas2_phys, inp_RijnMaas3_phys


    from Properties_networks.gegs_RMM_spec import inp_NW_NM_OM_1_geo , inp_NW_NM_OM_1_phys
    from Properties_networks.gegs_Hooghly import inp_hooghly1_geo, inp_hooghly2_geo, inp_hooghly3_geo, inp_hooghly1_phys
    from Properties_networks.gegs_HollandseIJssel import inp_HollandseIJssel1_geo,inp_HollandseIJssel1_phys
