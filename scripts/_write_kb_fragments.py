"""
One-time writer for conservation KB fragment JSON files.
Run: python scripts/_write_kb_fragments.py
"""
from __future__ import annotations

import json
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "databases" / "kb"


def write(name: str, data: dict) -> None:
    path = KB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")


SUBSTRATES = {
    "limestone_porous": {
        "id": "limestone_porous",
        "name": "Porous Limestone",
        "aliases": ["oolitic limestone", "calcarenite", "Ethiopian limestone", "Tigray limestone"],
        "composition": "Calcium carbonate (CaCO3) with intergranular porosity 15–35%; may contain clay seams, fossil fragments, and iron oxide inclusions",
        "properties": {
            "porosity_percent": "15-35",
            "hardness_mohs": "3-4",
            "permeability": "high",
            "ph": "alkaline",
            "salt_susceptibility": "very_high",
            "consolidant_penetration": "good"
        },
        "ethiopian_context": "Primary substrate for Aksumite stelae inscriptions, church facades in Lalibela and Gondar, and rock-hewn tomb reliefs. NGU Bulletin 436 documents extensive calcareous formations in central Ethiopia used historically for monumental carving.",
        "common_decay": ["salt_crystallization", "biological_colonization", "granular_disintegration", "black_crust", "biological_crust_inscription", "erosion_abrasion", "efflorescence", "contour_scaling"]
    },
    "limestone_dense": {
        "id": "limestone_dense",
        "name": "Dense Limestone",
        "aliases": ["compact limestone", "micritic limestone", "fine-grained limestone"],
        "composition": "Low-porosity micritic calcite matrix, porosity typically 3–10%",
        "properties": {
            "porosity_percent": "3-10",
            "hardness_mohs": "3-4",
            "permeability": "low",
            "ph": "alkaline",
            "salt_susceptibility": "moderate",
            "consolidant_penetration": "limited"
        },
        "ethiopian_context": "Used in select Aksumite architectural elements and dressed stone blocks where finer carving detail was required.",
        "common_decay": ["dissolution", "black_crust", "biological_colonization", "crack_mechanical", "soiling"]
    },
    "marble": {
        "id": "marble",
        "name": "Marble",
        "aliases": ["crystalline marble", "metamorphic carbonate"],
        "composition": "Recrystallized calcite/dolomite, grain size 0.1–2 mm, porosity 0.5–2%",
        "properties": {
            "porosity_percent": "0.5-2",
            "hardness_mohs": "3-4",
            "permeability": "very_low",
            "ph": "alkaline",
            "salt_susceptibility": "low_to_moderate",
            "polish_susceptibility": "high"
        },
        "ethiopian_context": "Rare in Ethiopian contexts; occasional imported marble elements in royal and ecclesiastical furnishings.",
        "common_decay": ["sugaring", "dissolution", "black_crust", "erosion_abrasion", "thermal_disaggregation"]
    },
    "sandstone": {
        "id": "sandstone",
        "name": "Sandstone",
        "aliases": ["arenite", "quartz sandstone", "feldspathic sandstone"],
        "composition": "Quartz and feldspar grains cemented by silica, calcite, or iron oxide; porosity 10–25%",
        "properties": {
            "porosity_percent": "10-25",
            "hardness_mohs": "6-7",
            "permeability": "moderate_to_high",
            "cement_type": "variable",
            "salt_susceptibility": "high_if_calcareous_cement"
        },
        "ethiopian_context": "Building stone in eastern escarpment regions; used in vernacular architecture and some church construction.",
        "common_decay": ["granular_disintegration", "sanding", "salt_crystallization", "erosion_abrasion", "alveolization", "contour_scaling"]
    },
    "basalt": {
        "id": "basalt",
        "name": "Basalt",
        "aliases": ["volcanic basalt", "trap basalt", "Ethiopian flood basalt"],
        "composition": "Fine-grained mafic volcanic rock: plagioclase, pyroxene, olivine; occasional vesicles and amygdules",
        "properties": {
            "porosity_percent": "1-8",
            "hardness_mohs": "5-6",
            "permeability": "low_to_moderate",
            "iron_content": "high",
            "weathering_mode": "exfoliation_and_mineral_alteration"
        },
        "ethiopian_context": "Dominant stone of the Ethiopian Highlands volcanic plateau. Aksumite stelae, Lalibela rock churches, and Gondar castle masonry are carved from local basalt and related volcanics.",
        "common_decay": ["spalling", "crack_mechanical", "biological_colonization", "iron_staining", "erosion_abrasion", "alveolization", "honeycombing"]
    },
    "tuff_ignimbrite": {
        "id": "tuff_ignimbrite",
        "name": "Tuff / Ignimbrite",
        "aliases": ["volcanic tuff", "welded tuff", "ignimbrite", "pumice tuff"],
        "composition": "Lithified volcanic ash and pumice fragments; highly variable porosity and cementation",
        "properties": {
            "porosity_percent": "20-45",
            "hardness_mohs": "3-5",
            "permeability": "high",
            "friability": "high",
            "salt_susceptibility": "very_high"
        },
        "ethiopian_context": "Common in Rift Valley and highland volcanic zones. Used in vernacular earthen-volcanic architecture and some carved elements.",
        "common_decay": ["capillary_rise_damage", "salt_crystallization", "granular_disintegration", "erosion_abrasion", "rising_damp", "efflorescence", "contour_scaling"]
    },
    "granite_granitoid": {
        "id": "granite_granitoid",
        "name": "Granite / Granitoid",
        "aliases": ["granite", "granodiorite", "syenite"],
        "composition": "Coarse-grained intrusive igneous rock: quartz, feldspar, mica; low porosity",
        "properties": {
            "porosity_percent": "0.5-2",
            "hardness_mohs": "6-7",
            "permeability": "very_low",
            "weathering_resistance": "high"
        },
        "ethiopian_context": "Precambrian basement complexes exposed in western and southern Ethiopia; occasional use in monumental architecture.",
        "common_decay": ["crack_mechanical", "biological_colonization", "erosion_abrasion", "soiling", "thermal_disaggregation"]
    },
    "terracotta_ceramic": {
        "id": "terracotta_ceramic",
        "name": "Terracotta / Ceramic",
        "aliases": ["fired clay", "earthenware", "pottery", "ceramic body"],
        "composition": "Fired clay body with variable temper (sand, grog, organic); glaze or slip may be present",
        "properties": {
            "porosity_percent": "10-25",
            "hardness_mohs": "5-7",
            "firing_temperature_c": "600-1100",
            "water_absorption": "moderate_to_high_if_unglazed"
        },
        "ethiopian_context": "Aksumite and post-Aksumite pottery traditions; church vessels, domestic ware, and archaeological ceramics from excavation contexts.",
        "common_decay": ["salt_crystallization", "crack_mechanical", "spalling", "previous_treatment_failure", "efflorescence", "soiling"]
    },
    "earthen_mud_brick": {
        "id": "earthen_mud_brick",
        "name": "Earthen / Mud Brick",
        "aliases": ["adobe", "mud brick", "rammed earth", "cob"],
        "composition": "Clay, silt, sand, organic binder; often unfired or low-fired",
        "properties": {
            "porosity_percent": "25-40",
            "hardness_mohs": "1-2",
            "permeability": "very_high",
            "water_sensitivity": "extreme"
        },
        "ethiopian_context": "Harar Jugol walled city, vernacular highland housing, and earthen church structures. ICOMOS-ISCEAH earthen deterioration patterns apply.",
        "common_decay": ["capillary_rise_damage", "rising_damp", "erosion_abrasion", "crack_mechanical", "salt_crystallization", "efflorescence", "humidity_damage"]
    },
    "parchment_vellum": {
        "id": "parchment_vellum",
        "name": "Parchment / Vellum",
        "aliases": ["parchment", "vellum", "animal skin manuscript"],
        "composition": "Degreased and limed animal skin (sheep, goat, calf); collagen matrix with residual lipids",
        "properties": {
            "ph": "neutral_to_slightly_alkaline_if_well_prepared",
            "humidity_range_percent": "45-55",
            "light_sensitivity": "high",
            "mechanical_strength": "moderate_when_dry_brittle_when_desiccated"
        },
        "ethiopian_context": "Ethiopian manuscript tradition (Ge'ez texts) on parchment; church and monastic library holdings.",
        "common_decay": ["delamination_parchment", "ink_corrosion", "mould_biological_organic", "humidity_damage", "insect_infestation", "warping_wood"]
    },
    "wood": {
        "id": "wood",
        "name": "Wood",
        "aliases": ["timber", "wooden artefact", "ligneous material"],
        "composition": "Cellulose, hemicellulose, lignin; species-dependent density and extractives",
        "properties": {
            "humidity_range_percent": "45-55",
            "anisotropic_shrinkage": True,
            "biodegradability": "high",
            "light_sensitivity": "moderate"
        },
        "ethiopian_context": "Church doors, processional crosses, tabots (replica), furniture, and architectural elements in historic structures.",
        "common_decay": ["fungal_decay_wood", "insect_infestation", "warping_wood", "crack_mechanical", "mould_biological_organic", "humidity_damage"]
    },
    "metal_bronze": {
        "id": "metal_bronze",
        "name": "Bronze",
        "aliases": ["copper alloy", "bronze coin", "brass"],
        "composition": "Copper-tin alloy (typically 88% Cu, 12% Sn); may include lead, zinc; surface patina of cuprite, malachite, atacamite",
        "properties": {
            "corrosion_products": "cuprite_malachite_atacamite",
            "chloride_sensitivity": "very_high",
            "stable_patina_possible": True
        },
        "ethiopian_context": "Aksumite coins, church bells, processional crosses, and metal fittings.",
        "common_decay": ["bronze_disease", "pitting_corrosion", "patina_unstable", "soiling", "previous_treatment_failure"]
    },
    "metal_iron_steel": {
        "id": "metal_iron_steel",
        "name": "Iron / Steel",
        "aliases": ["wrought iron", "cast iron", "ferrous metal"],
        "composition": "Iron-carbon alloy; corrosion products: goethite, lepidocrocite, magnetite",
        "properties": {
            "corrosion_rate": "high_in_humid_chloride_environments",
            "galvanic_risk_with_copper_alloys": True
        },
        "ethiopian_context": "Nails, tools, weapons, and architectural ironwork in historic buildings.",
        "common_decay": ["rust_corrosion", "pitting_corrosion", "iron_staining", "previous_treatment_failure"]
    },
    "painted_surface_fresco": {
        "id": "painted_surface_fresco",
        "name": "Painted Surface / Fresco",
        "aliases": ["fresco", "wall painting", "polychrome surface", "tempera on plaster"],
        "composition": "Pigment layer (earth pigments, carbon black, orpiment) bound in lime, casein, or animal glue on plaster or stone support",
        "properties": {
            "layer_structure": "pigment_binder_ground_support",
            "light_sensitivity": "high",
            "solvent_sensitivity": "variable_by_binder"
        },
        "ethiopian_context": "Church interior murals (Gondar, Lake Tana monasteries), painted stelae fragments, and decorated architectural surfaces.",
        "common_decay": ["paint_detachment", "flaking", "efflorescence_paint", "biological_colonization", "humidity_damage", "soiling"]
    },
    "ivory_bone": {
        "id": "ivory_bone",
        "name": "Ivory / Bone",
        "aliases": ["ivory", "bone", "antler", "osseous material"],
        "composition": "Collagen-hydroxyapatite composite; ivory is dense dentine, bone is more porous",
        "properties": {
            "humidity_range_percent": "45-55",
            "light_sensitivity": "high",
            "organic_sensitivity": "extreme"
        },
        "ethiopian_context": "Carved crosses, inlay work, and liturgical objects; archaeological bone artefacts.",
        "common_decay": ["crack_mechanical", "humidity_damage", "mould_biological_organic", "insect_infestation", "warping_wood"]
    },
    "schist": {
        "id": "schist",
        "name": "Schist",
        "aliases": ["metamorphic schist", "foliated schist"],
        "composition": "Foliated metamorphic rock with mica, quartz, feldspar; pronounced cleavage planes",
        "properties": {
            "foliation": "pronounced",
            "hardness_mohs": "4-6",
            "delamination_risk": "high"
        },
        "ethiopian_context": "Precambrian metamorphic terranes; local building stone in some highland areas.",
        "common_decay": ["delamination_parchment", "contour_scaling", "erosion_abrasion", "biological_colonization", "freeze_thaw"]
    },
    "slate": {
        "id": "slate",
        "name": "Slate",
        "aliases": ["slate stone", "fine-grained slate"],
        "composition": "Low-grade metamorphic shale-derived rock with strong cleavage",
        "properties": {
            "cleavage": "excellent",
            "hardness_mohs": "5-6",
            "porosity_percent": "1-3"
        },
        "ethiopian_context": "Limited use; occasional roofing and paving in historic contexts.",
        "common_decay": ["contour_scaling", "erosion_abrasion", "freeze_thaw", "soiling"]
    },
    "quartzite": {
        "id": "quartzite",
        "name": "Quartzite",
        "aliases": ["metamorphic quartzite", "siliceous sandstone"],
        "composition": "Recrystallized quartz grains, very low porosity",
        "properties": {
            "porosity_percent": "1-3",
            "hardness_mohs": "7",
            "acid_resistance": "high"
        },
        "ethiopian_context": "Hard durable stone in basement complex regions; occasional monumental use.",
        "common_decay": ["erosion_abrasion", "thermal_disaggregation", "crack_mechanical", "soiling"]
    },
    "travertine": {
        "id": "travertine",
        "name": "Travertine",
        "aliases": ["calcium carbonate tufa", "freshwater limestone"],
        "composition": "CaCO3 precipitated from groundwater; high macroporosity with vughs and channels",
        "properties": {
            "porosity_percent": "10-30",
            "hardness_mohs": "3-4",
            "permeability": "high"
        },
        "ethiopian_context": "Limited occurrence; thermal spring deposits in Rift Valley.",
        "common_decay": ["dissolution", "salt_crystallization", "alveolization", "honeycombing", "biological_colonization"]
    },
    "plaster_lime": {
        "id": "plaster_lime",
        "name": "Lime Plaster",
        "aliases": ["lime plaster", "lime mortar render", "haired lime plaster"],
        "composition": "Calcium hydroxide/carbonate binder with sand aggregate; may include hair or plant fibres",
        "properties": {
            "breathability": "high",
            "carbonation_required": True,
            "ph": "alkaline"
        },
        "ethiopian_context": "Interior and exterior renders on stone and earthen architecture; church wall preparations for murals.",
        "common_decay": ["crack_mechanical", "efflorescence", "rising_damp", "capillary_rise_damage", "paint_detachment", "humidity_damage"]
    },
    "stucco": {
        "id": "stucco",
        "name": "Stucco",
        "aliases": ["decorative stucco", "lime stucco", "gesso"],
        "composition": "Lime or gypsum-based decorative surface layer, often multi-coat with fine finish",
        "properties": {
            "layer_thickness_mm": "5-30",
            "water_sensitivity": "moderate",
            "decorative_fragility": "high"
        },
        "ethiopian_context": "Decorative architectural elements in Gondar-era buildings.",
        "common_decay": ["flaking", "crack_mechanical", "humidity_damage", "efflorescence", "biological_colonization"]
    },
    "glass": {
        "id": "glass",
        "name": "Glass",
        "aliases": ["soda-lime glass", "historic glass", "vitreous material"],
        "composition": "Silica network with soda and lime flux; corrosion layer (cristobalite, alkalis leached)",
        "properties": {
            "cristallinity": "amorphous",
            "weathering_type": "alkali_leaching",
            "humidity_sensitivity": "high"
        },
        "ethiopian_context": "Rare in Ethiopian heritage; occasional church window panes and imported vessels.",
        "common_decay": ["crack_mechanical", "humidity_damage", "accretion_deposits", "soiling"]
    },
    "leather": {
        "id": "leather",
        "name": "Leather",
        "aliases": ["tanned leather", "parchment binding leather", "bookbinding leather"],
        "composition": "Collagen matrix cross-linked by tanning agents (vegetable, alum, or chrome)",
        "properties": {
            "humidity_range_percent": "45-55",
            "red_rot_risk": "high_for_sulfur_tanned",
            "light_sensitivity": "high"
        },
        "ethiopian_context": "Manuscript bindings, belts, and liturgical accessories.",
        "common_decay": ["mould_biological_organic", "humidity_damage", "insect_infestation", "crack_mechanical"]
    },
    "textile_linen": {
        "id": "textile_linen",
        "name": "Textile / Linen",
        "aliases": ["linen", "cotton textile", "wool fabric", "woven textile"],
        "composition": "Cellulosic (linen, cotton) or proteinaceous (wool, silk) fibres",
        "properties": {
            "humidity_range_percent": "45-55",
            "light_sensitivity": "high",
            "pest_susceptibility": "high"
        },
        "ethiopian_context": "Liturgical vestments, shrouds, and ecclesiastical textiles.",
        "common_decay": ["mould_biological_organic", "insect_infestation", "humidity_damage", "soiling", "crack_mechanical"]
    },
    "paper": {
        "id": "paper",
        "name": "Paper",
        "aliases": ["handmade paper", "rag paper", "wood-pulp paper"],
        "composition": "Cellulose fibres; sizing (gelatin, alum) and fillers may be present",
        "properties": {
            "ph": "acidic_if_wood_pulp",
            "humidity_range_percent": "45-55",
            "light_sensitivity": "very_high"
        },
        "ethiopian_context": "Modern manuscript copies and archival documents; limited historic paper tradition.",
        "common_decay": ["mould_biological_organic", "humidity_damage", "ink_corrosion", "insect_infestation", "crack_mechanical"]
    },
    "mosaic_tesserae": {
        "id": "mosaic_tesserae",
        "name": "Mosaic Tesserae",
        "aliases": ["mosaic", "tesserae", "opus sectile"],
        "composition": "Small stone, glass, or ceramic pieces set in lime or cement mortar bed",
        "properties": {
            "composite_nature": True,
            "mortar_bed_sensitivity": "high",
            "intervention_complexity": "high"
        },
        "ethiopian_context": "Limited mosaic tradition; occasional floor and decorative panels.",
        "common_decay": ["crack_mechanical", "efflorescence", "previous_treatment_failure", "soiling", "paint_detachment"]
    },
    "gilded_surface": {
        "id": "gilded_surface",
        "name": "Gilded Surface",
        "aliases": ["gold leaf", "gilding", "vermeil"],
        "composition": "Gold leaf (0.1–0.2 µm) over bole/clay ground on wood, stone, or metal support",
        "properties": {
            "layer_thickness_um": "0.1-0.2",
            "mechanical_fragility": "extreme",
            "cleaning_restriction": "no_abrasive_or_solvent"
        },
        "ethiopian_context": "Gilded processional crosses, icon panels, and church furnishing.",
        "common_decay": ["flaking", "crack_mechanical", "humidity_damage", "soiling", "previous_treatment_failure"]
    },
    "dolomite": {
        "id": "dolomite",
        "name": "Dolomite",
        "aliases": ["dolomitic limestone", "magnesian limestone"],
        "composition": "CaMg(CO3)2; intermediate between limestone and marble",
        "properties": {
            "porosity_percent": "5-15",
            "hardness_mohs": "3.5-4",
            "acid_sensitivity": "high"
        },
        "ethiopian_context": "Occasional use in carbonate formations of the Ethiopian plateau.",
        "common_decay": ["dissolution", "salt_crystallization", "granular_disintegration", "black_crust"]
    },
    "conglomerate": {
        "id": "conglomerate",
        "name": "Conglomerate",
        "aliases": ["puddingstone", "breccia matrix", "cemented conglomerate"],
        "composition": "Rounded clasts in siliceous or calcareous cement matrix",
        "properties": {
            "heterogeneity": "high",
            "differential_weathering": "pronounced",
            "porosity_percent": "5-20"
        },
        "ethiopian_context": "Sedimentary formations in rift margins; occasional building stone.",
        "common_decay": ["granular_disintegration", "erosion_abrasion", "contour_scaling", "alveolization"]
    },
    "gypsum_alabaster": {
        "id": "gypsum_alabaster",
        "name": "Gypsum / Alabaster",
        "aliases": ["alabaster", "gypsum", "selenite"],
        "composition": "Hydrated calcium sulfate CaSO4·2H2O; very soft and water-soluble",
        "properties": {
            "hardness_mohs": "2",
            "water_solubility": "high",
            "porosity_percent": "5-15"
        },
        "ethiopian_context": "Rare; evaporite deposits in eastern lowlands.",
        "common_decay": ["dissolution", "humidity_damage", "crack_mechanical", "soiling"]
    },
    "obsidian": {
        "id": "obsidian",
        "name": "Obsidian",
        "aliases": ["volcanic glass", "natural glass"],
        "composition": "Amorphous silica-rich volcanic glass; conchoidal fracture",
        "properties": {
            "hardness_mohs": "5-6",
            "fracture": "conchoidal",
            "weathering_resistance": "high_when_stable"
        },
        "ethiopian_context": "Ethiopian Rift Valley obsidian sources; prehistoric tools and occasional ritual objects.",
        "common_decay": ["crack_mechanical", "dissolution", "soiling"]
    },
    "laterite": {
        "id": "laterite",
        "name": "Laterite",
        "aliases": ["lateritic stone", "iron-cemented laterite"],
        "composition": "Iron and aluminium oxides cementing clay matrix; tropical weathering product",
        "properties": {
            "iron_content": "very_high",
            "porosity_percent": "15-30",
            "hardness_when_dry": "moderate",
            "softening_when_wet": True
        },
        "ethiopian_context": "Western lowland and Sudan-border regions; vernacular architecture material.",
        "common_decay": ["erosion_abrasion", "humidity_damage", "capillary_rise_damage", "salt_crystallization", "crack_mechanical"]
    },
}

DECAY_PATTERNS = {
    "salt_crystallization": {
        "id": "salt_crystallization",
        "name": "Salt Crystallization",
        "icomos_classification": "C.5.1 Crystallization of salts (subflorescence and efflorescence)",
        "aliases": ["salt weathering", "salt attack", "haloclasty"],
        "description": "Crystallization pressure from hygroscopic salts (NaCl, Na2SO4, MgSO4, CaCl2) within pores exceeds tensile strength of stone matrix, causing granular disintegration and scaling. Subflorescence (beneath surface) is more destructive than efflorescence (surface crust).",
        "detection_method": "Visual white crystalline deposits; conductivity measurement; ion chromatography of poultice extracts; UV fluorescence of nitrate salts",
        "axum_detection": "salt_mapper UV fluorescence + conductivity probe; elevated salt_risk score ≥ 0.15",
        "urgency_factors": ["active moisture source", "inscription present", "subflorescence confirmed", "cyclic RH fluctuation"],
        "affected_substrates": ["limestone_porous", "limestone_dense", "sandstone", "tuff_ignimbrite", "terracotta_ceramic", "earthen_mud_brick", "plaster_lime", "dolomite", "laterite", "travertine", "conglomerate", "mosaic_tesserae"]
    },
    "efflorescence": {
        "id": "efflorescence",
        "name": "Efflorescence",
        "icomos_classification": "C.5.1.1 Efflorescence",
        "aliases": ["surface salt bloom", "white bloom"],
        "description": "Salts crystallize on the external surface forming white powdery or fluffy deposits. Indicates active salt migration; precursor to subflorescence damage.",
        "detection_method": "Visual inspection; tape lift microscopy; conductivity of dissolved surface salts",
        "axum_detection": "salt_mapper fluorescence; salt_risk ≥ 0.15",
        "urgency_factors": ["recurring after cleaning", "associated with rising damp", "on painted surface"],
        "affected_substrates": ["limestone_porous", "sandstone", "tuff_ignimbrite", "terracotta_ceramic", "earthen_mud_brick", "plaster_lime", "brick", "laterite"]
    },
    "subflorescence": {
        "id": "subflorescence",
        "name": "Subflorescence",
        "icomos_classification": "C.5.1.2 Subflorescence",
        "aliases": ["cryptoflorescence", "sub-surface salt crystallization"],
        "description": "Salt crystallization beneath the surface causes contour scaling, blistering, and detachment without visible surface deposit. Most destructive salt weathering mechanism per ICOMOS-ISCS.",
        "detection_method": "IR thermography (delamination zones); ultrasonic pulse velocity reduction; destructive sampling of scaling layers",
        "axum_detection": "salt_critical flag + stress_score elevation; multispectral NDCI stress mapping",
        "urgency_factors": ["scaling active", "inscription legibility threatened", "salt_critical sensor flag"],
        "affected_substrates": ["limestone_porous", "sandstone", "tuff_ignimbrite", "terracotta_ceramic", "plaster_lime", "dolomite", "travertine"]
    },
    "biological_colonization": {
        "id": "biological_colonization",
        "name": "Biological Colonization",
        "icomos_classification": "B. Biological colonization",
        "aliases": ["biopatina", "lichen", "algae", "bryophyte", "microbial mat"],
        "description": "Colonisation by algae, lichens, mosses, fungi, and bacteria on damp stone surfaces. Biochemical activity produces organic acids, chelating agents, and mechanical rootlet penetration.",
        "detection_method": "Visual green/black/orange staining; chlorophyll fluorescence; ATP bioluminescence assay",
        "axum_detection": "multispectral biological classifier; biological_detected sensor flag",
        "urgency_factors": ["covering inscription", "active moisture", "south-facing damp exposure"],
        "affected_substrates": ["limestone_porous", "limestone_dense", "sandstone", "basalt", "marble", "granite_granitoid", "tuff_ignimbrite", "plaster_lime", "stucco", "painted_surface_fresco", "schist", "travertine", "conglomerate"]
    },
    "crack_mechanical": {
        "id": "crack_mechanical",
        "name": "Mechanical Cracking",
        "icomos_classification": "A.1 Structural damage — cracks",
        "aliases": ["structural crack", "fissure", "fracture", "hairline crack"],
        "description": "Fractures from structural loading, thermal stress, root penetration, impact, or previous intervention. May propagate with moisture and freeze-thaw cycling.",
        "detection_method": "Visual survey; crack width measurement (feeler gauge); acoustic tap (void detection); photometric stereo stress mapping",
        "axum_detection": "crack detector severity score ≥ 0.25",
        "urgency_factors": ["propagating", "through-thickness", "on inscription zone", "associated with detachment"],
        "affected_substrates": ["limestone_porous", "limestone_dense", "marble", "sandstone", "basalt", "granite_granitoid", "terracotta_ceramic", "earthen_mud_brick", "wood", "painted_surface_fresco", "plaster_lime", "stucco", "glass", "ivory_bone", "schist", "obsidian", "mosaic_tesserae", "gilded_surface"]
    },
    "spalling": {
        "id": "spalling",
        "name": "Spalling",
        "icomos_classification": "A.2 Surface damage — spalling",
        "aliases": ["exfoliation", "flaking stone", "surface loss"],
        "description": "Detachment of surface layers in plates or scales. Caused by salt crystallization, freeze-thaw, fire damage, or incompatible surface treatments.",
        "detection_method": "Visual sounding (tap test); IR thermography; measured surface recession",
        "axum_detection": "stress_score ≥ 0.65; crack detector combined with salt_risk",
        "urgency_factors": ["active loss", "inscription affected", "overhanging spall (safety)"],
        "affected_substrates": ["basalt", "granite_granitoid", "sandstone", "limestone_porous", "tuff_ignimbrite", "terracotta_ceramic", "schist", "conglomerate"]
    },
    "granular_disintegration": {
        "id": "granular_disintegration",
        "name": "Granular Disintegration",
        "icomos_classification": "C.1 Disintegration — granular disintegration",
        "aliases": ["powdering", "sugaring precursor", "surface sanding"],
        "description": "Individual grains or crystals detach from the surface leaving a rough, sugary texture. Caused by salt weathering, acid rain, or biological acids.",
        "detection_method": "Surface roughness measurement; grain loss quantification (cellophane tape method); hardness probe",
        "axum_detection": "stress_score ≥ 0.45 or hardness_score < 0.35",
        "urgency_factors": ["inscription legibility loss", "active salt migration", "accelerating powdering rate"],
        "affected_substrates": ["limestone_porous", "sandstone", "marble", "dolomite", "tuff_ignimbrite", "travertine", "conglomerate", "laterite"]
    },
    "dissolution": {
        "id": "dissolution",
        "name": "Dissolution",
        "icomos_classification": "C.2 Dissolution",
        "aliases": ["acid dissolution", "karstification", "surface etching"],
        "description": "Chemical dissolution of carbonate or sulfate matrix by acidic rain, organic acids from biocolonisation, or cleaning agents. Produces pitting and loss of surface detail.",
        "detection_method": "Surface pH measurement; runoff water analysis; micro-erosion meter",
        "axum_detection": "low hardness_score on carbonate substrates; multispectral stress on etched zones",
        "urgency_factors": ["inscription detail loss", "active acid source", "protected zone undercutting"],
        "affected_substrates": ["limestone_porous", "limestone_dense", "marble", "dolomite", "travertine", "gypsum_alabaster", "plaster_lime"]
    },
    "black_crust": {
        "id": "black_crust",
        "name": "Black Crust",
        "icomos_classification": "D.1 Deposition — black crust",
        "aliases": ["gypsum crust", "anthropogenic crust", "soot crust"],
        "description": "Dark sulphate-rich crust from gypsum recrystallization trapping atmospheric particulates (soot, fly ash). Common on sheltered carbonate surfaces in urban environments.",
        "detection_method": "Visual; SEM-EDS for gypsum and carbon; cross-section microscopy",
        "axum_detection": "multispectral dark crust detection; OCR obscuration on sheltered faces",
        "urgency_factors": ["obscuring inscription", "trapping moisture beneath crust", "acid-generating crust"],
        "affected_substrates": ["limestone_porous", "limestone_dense", "marble", "sandstone", "dolomite", "plaster_lime"]
    },
    "biological_crust_inscription": {
        "id": "biological_crust_inscription",
        "name": "Biological Crust on Inscription",
        "icomos_classification": "B. Biological colonization (specialised: inscription zone)",
        "aliases": ["inscription biofilm", "lichen on text", "algal film on carving"],
        "description": "Biological mat specifically obscuring carved or painted inscription zones. Low OCR confidence is a proxy indicator in AXUM pipeline.",
        "detection_method": "OCR legibility assessment; magnification; chlorophyll fluorescence on text zones",
        "axum_detection": "has_inscription + ocr_confidence < 0.55 + biological_detected",
        "urgency_factors": ["Ge'ez text obscured", "active growth", "acid-producing lichen species"],
        "affected_substrates": ["limestone_porous", "basalt", "sandstone", "painted_surface_fresco", "parchment_vellum"]
    },
    "iron_staining": {
        "id": "iron_staining",
        "name": "Iron Staining",
        "icomos_classification": "D.2 Deposition — iron staining",
        "aliases": ["rust stain", "iron oxide stain", "metallic stain"],
        "description": "Orange-brown iron oxide/hydroxide deposits from corroding iron fixings, groundwater iron, or iron-rich mineral inclusions oxidizing.",
        "detection_method": "Visual colour assessment; XRF for Fe; chemical test (potassium ferricyanide)",
        "axum_detection": "multispectral iron oxide band detection",
        "urgency_factors": ["spreading stain", "associated active iron corrosion", "on light-coloured inscription"],
        "affected_substrates": ["limestone_porous", "marble", "sandstone", "plaster_lime", "parchment_vellum", "textile_linen"]
    },
    "previous_treatment_failure": {
        "id": "previous_treatment_failure",
        "name": "Previous Treatment Failure",
        "icomos_classification": "E. Human intervention — harmful treatment",
        "aliases": ["incompatible past treatment", "failed consolidation", "discoloured repair"],
        "description": "Damage from earlier conservation or restoration using incompatible materials (Portland cement, acrylic sealants, hard mortars, improper cleaning).",
        "detection_method": "UV fluorescence of synthetic polymers; mortar analysis (petrography); documentary research",
        "axum_detection": "visual inspection flag; hardness anomalies; synthetic fluorescence under UV",
        "urgency_factors": ["active detachment at repair boundary", "trapped moisture", "accelerating decay at interface"],
        "affected_substrates": ["limestone_porous", "sandstone", "basalt", "terracotta_ceramic", "plaster_lime", "painted_surface_fresco", "metal_bronze", "mosaic_tesserae", "gilded_surface"]
    },
    "erosion_abrasion": {
        "id": "erosion_abrasion",
        "name": "Erosion / Abrasion",
        "icomos_classification": "C.3 Erosion",
        "aliases": ["wind erosion", "sand abrasion", "surface wear", "rounded detail loss"],
        "description": "Progressive surface loss from wind-driven particles, water flow, human touch, or improper cleaning. Inscriptions lose depth and legibility.",
        "detection_method": "Surface recession measurement; micro-erosion meter; comparison with archival photographs",
        "axum_detection": "hardness_score reduction; photometric stereo depth loss",
        "urgency_factors": ["inscription depth < 1mm", "exposed position", "continued abrasive contact"],
        "affected_substrates": ["limestone_porous", "sandstone", "basalt", "tuff_ignimbrite", "earthen_mud_brick", "laterite", "schist", "conglomerate", "marble"]
    },
    "mould_biological_organic": {
        "id": "mould_biological_organic",
        "name": "Mould / Biological Decay (Organic)",
        "icomos_classification": "B. Biological colonization (organic substrates)",
        "aliases": ["mould", "foxing", "fox spots", "fungal growth"],
        "description": "Fungal colonisation of organic materials producing staining, weakening, and allergenic spores. Thrives above 65% RH.",
        "detection_method": "Visual foxing spots; UV fluorescence (some moulds); ATP assay; culture (destructive)",
        "axum_detection": "biological_detected + organic substrate class; active_moisture flag",
        "urgency_factors": ["RH > 65%", "active growth", "valuable manuscript", "insect attraction"],
        "affected_substrates": ["parchment_vellum", "wood", "paper", "textile_linen", "leather", "ivory_bone"]
    },
    "bronze_disease": {
        "id": "bronze_disease",
        "name": "Bronze Disease",
        "icomos_classification": "C.5.2 Metal corrosion — active bronze disease",
        "aliases": ["cuprous chloride corrosion", "active corrosion", "powdery green corrosion"],
        "description": "Cyclic corrosion driven by cuprous chloride (nantokite) in presence of moisture and oxygen. Produces powdery light-green spots that reappear after cleaning.",
        "detection_method": "Visual powdery green spots; XRD for nantokite/atacamite; chloride ion detection",
        "axum_detection": "salt_risk ≥ 0.1 on metal_bronze substrate; powdery corrosion visual flag",
        "urgency_factors": ["powdering active", "high chloride", "cyclic RH", "post-excavation exposure"],
        "affected_substrates": ["metal_bronze"]
    },
    "delamination_parchment": {
        "id": "delamination_parchment",
        "name": "Delamination (Parchment / Foliated Stone)",
        "icomos_classification": "A.2 Surface damage — delamination",
        "aliases": ["layer separation", "foliation", "lamination failure"],
        "description": "Separation along weak planes in parchment (gelatin layer) or foliated stone (schist, slate). Caused by RH cycling, mechanical stress, or adhesive failure.",
        "detection_method": "Raking light examination; ultrasonic C-scan; mechanical probing (careful)",
        "axum_detection": "ocr_confidence drop on parchment; stress_score on foliated stone",
        "urgency_factors": ["active separation", "text on delaminating layer", "handling planned"],
        "affected_substrates": ["parchment_vellum", "schist", "slate", "gilded_surface"]
    },
    "contour_scaling": {
        "id": "contour_scaling",
        "name": "Contour Scaling",
        "icomos_classification": "A.2 Surface damage — contour scaling",
        "aliases": ["scaling", "subflorescence scaling", "blister scaling"],
        "description": "Thin scales of stone detach following the contour of the surface. Primary mechanism is subflorescence salt crystallization pressure.",
        "detection_method": "Sounding; detachment survey; salt analysis of scaled material",
        "axum_detection": "salt_risk + stress_score combination on porous stone",
        "urgency_factors": ["active scaling", "salt migration ongoing", "inscription on scaled layer"],
        "affected_substrates": ["limestone_porous", "sandstone", "tuff_ignimbrite", "dolomite", "schist", "conglomerate", "plaster_lime"]
    },
    "ink_corrosion": {
        "id": "ink_corrosion",
        "name": "Ink Corrosion",
        "icomos_classification": "C.5.3 Other — ink corrosion",
        "aliases": ["iron gall ink corrosion", "ink burn", "ink degradation"],
        "description": "Iron gall inks oxidize and produce sulfuric acid that embrittles and burns through parchment or paper support. Common in historic manuscripts.",
        "detection_method": "Transmitted light (transparency around ink); pH mapping; iron test on ink lines",
        "axum_detection": "ocr_confidence < 0.5 on parchment with inscription",
        "urgency_factors": ["holes forming at ink lines", "brittle zone expanding", "handling scheduled"],
        "affected_substrates": ["parchment_vellum", "paper"]
    },
    "sugaring": {
        "id": "sugaring",
        "name": "Sugaring",
        "icomos_classification": "C.1 Disintegration — sugaring",
        "aliases": ["marble sugaring", "crystalline disintegration"],
        "description": "Individual calcite crystals protrude and detach on marble and crystalline limestone, producing a sparkling granular surface. Caused by repeated dissolution-reprecipitation.",
        "detection_method": "Visual granular sparkling surface; SEM of detached grains",
        "axum_detection": "hardness_score reduction on marble/crystalline carbonate",
        "urgency_factors": ["polished surface loss", "sculptural detail rounding", "outdoor exposure"],
        "affected_substrates": ["marble", "limestone_dense", "dolomite"]
    },
    "sanding": {
        "id": "sanding",
        "name": "Sanding",
        "icomos_classification": "C.1 Disintegration — sanding",
        "aliases": ["sand erosion texture", "aeolian sanding"],
        "description": "Surface wears to a smooth sandpaper-like texture from wind-driven sand abrasion. Common on exposed sandstone.",
        "detection_method": "Surface texture profiling; comparison photography",
        "axum_detection": "erosion pattern on sandstone; hardness reduction",
        "urgency_factors": ["inscription depth loss", "continued sand exposure"],
        "affected_substrates": ["sandstone", "tuff_ignimbrite", "laterite", "earthen_mud_brick"]
    },
    "flaking": {
        "id": "flaking",
        "name": "Flaking",
        "icomos_classification": "A.2 Surface damage — flaking",
        "aliases": ["paint flake", "gilding flake", "plaster flake"],
        "description": "Small flat fragments detach from painted, gilded, or plastered surfaces due to adhesion failure, moisture, or salt behind the layer.",
        "detection_method": "Raking light; magnification; adhesion test (tape — use with caution)",
        "axum_detection": "stress_score on painted/gilded surfaces",
        "urgency_factors": ["active flake loss", "original pigment layer", "moisture behind paint"],
        "affected_substrates": ["painted_surface_fresco", "gilded_surface", "stucco", "plaster_lime", "wood"]
    },
    "paint_detachment": {
        "id": "paint_detachment",
        "name": "Paint Detachment",
        "icomos_classification": "A.2 Surface damage — detachment (paint layer)",
        "aliases": ["blistering paint", "paint loss", "mural detachment"],
        "description": "Paint or pigment layer separates from ground or support in sheets or blisters. Caused by moisture, salt, or incompatible overpaint.",
        "detection_method": "IR thermography; sounding; detachment mapping",
        "axum_detection": "stress_score ≥ 0.5 on painted_surface_fresco",
        "urgency_factors": ["large detached zones", "original pigment", "active moisture behind"],
        "affected_substrates": ["painted_surface_fresco", "stucco", "plaster_lime", "wood", "terracotta_ceramic"]
    },
    "capillary_rise_damage": {
        "id": "capillary_rise_damage",
        "name": "Capillary Rise Damage",
        "icomos_classification": "C.5.4 Moisture — capillary rise",
        "aliases": ["rising damp damage", "tide mark decay", "groundwater rise"],
        "description": "Moisture rises through porous materials by capillary action, depositing salts at the evaporation front and causing decay in the damp zone.",
        "detection_method": "Moisture meter profiling; salt analysis at tide mark; IR thermography",
        "axum_detection": "active_moisture on tuff_ignimbrite or earthen_mud_brick",
        "urgency_factors": ["tide mark rising", "salt accumulation at front", "structural base affected"],
        "affected_substrates": ["tuff_ignimbrite", "earthen_mud_brick", "limestone_porous", "sandstone", "plaster_lime", "laterite", "brick"]
    },
    "freeze_thaw": {
        "id": "freeze_thaw",
        "name": "Freeze-Thaw Cycling",
        "icomos_classification": "C.5.5 Moisture — freeze-thaw",
        "aliases": ["frost damage", "ice crystallization damage"],
        "description": "Water in pores expands 9% on freezing, generating hydraulic pressure that microcracks and spalls stone. Critical in high-altitude Ethiopian sites.",
        "detection_method": "Climate data correlation; crack mapping after winter; porosity assessment",
        "axum_detection": "crack_severity + environmental data (if available)",
        "urgency_factors": ["high altitude site", "saturated pores before freeze", "repeated cycling"],
        "affected_substrates": ["limestone_porous", "sandstone", "tuff_ignimbrite", "terracotta_ceramic", "plaster_lime", "schist", "slate", "conglomerate"]
    },
    "thermal_disaggregation": {
        "id": "thermal_disaggregation",
        "name": "Thermal Disaggregation",
        "icomos_classification": "C.4 Thermal disaggregation",
        "aliases": ["thermal shock", "fire damage", "insolation damage"],
        "description": "Rapid temperature change causes differential expansion and microcracking. Fire produces dramatic spalling; insolation causes daily cycling damage.",
        "detection_method": "Fire damage survey; crack orientation analysis; colour change (fire)",
        "axum_detection": "crack pattern analysis; discolouration detection",
        "urgency_factors": ["post-fire", "dark stone in hot sun", "recent thermal event"],
        "affected_substrates": ["basalt", "granite_granitoid", "marble", "quartzite", "sandstone", "obsidian"]
    },
    "pitting_corrosion": {
        "id": "pitting_corrosion",
        "name": "Pitting Corrosion",
        "icomos_classification": "C.5.2 Metal corrosion — pitting",
        "aliases": ["localized corrosion", "etch pitting"],
        "description": "Localized metal loss forming cavities. On bronze, associated with chloride ions; on iron, with differential aeration cells.",
        "detection_method": "Visual pit mapping; profilometry; metallographic section",
        "axum_detection": "surface pit detection on metal substrates",
        "urgency_factors": ["through-thickness pits", "structural element", "active chloride"],
        "affected_substrates": ["metal_bronze", "metal_iron_steel"]
    },
    "rust_corrosion": {
        "id": "rust_corrosion",
        "name": "Rust / Iron Corrosion",
        "icomos_classification": "C.5.2 Metal corrosion — rust",
        "aliases": ["iron rust", "ferrous corrosion", "oxidation"],
        "description": "Electrochemical oxidation of iron producing voluminous rust that exfoliates and stains adjacent materials. Rust occupies 6–8× original iron volume.",
        "detection_method": "Visual orange-brown corrosion; chloride test; corrosion rate monitoring",
        "axum_detection": "default on metal_iron_steel substrate",
        "urgency_factors": ["active rusting", "structural load-bearing", "staining adjacent stone"],
        "affected_substrates": ["metal_iron_steel"]
    },
    "insect_infestation": {
        "id": "insect_infestation",
        "name": "Insect Infestation",
        "icomos_classification": "B. Biological — insect damage",
        "aliases": ["woodworm", "bookworm", "beetle damage", "termite damage"],
        "description": "Larval boring by Anobiidae (furniture beetle), Lyctidae (powderpost), and termites in wood, leather, and paper. Exit holes and frass are diagnostic.",
        "detection_method": "Exit hole survey; frass detection; acoustic emission; X-ray (museum context)",
        "axum_detection": "organic substrate + biological_detected",
        "urgency_factors": ["active frass", "structural timber", "manuscript collection proximity"],
        "affected_substrates": ["wood", "parchment_vellum", "paper", "textile_linen", "leather", "ivory_bone"]
    },
    "fungal_decay_wood": {
        "id": "fungal_decay_wood",
        "name": "Fungal Decay (Wood)",
        "icomos_classification": "B. Biological — fungal decay",
        "aliases": ["dry rot", "wet rot", "brown rot", "white rot"],
        "description": "Basidiomycete fungi digest lignin and cellulose, reducing wood to brittle, cracked, or cuboidal fragments. Dry rot (Serpula) can spread through masonry.",
        "detection_method": "Visual cuboidal cracking; moisture content > 20%; mycological identification",
        "axum_detection": "biological_detected on wood + active_moisture",
        "urgency_factors": ["structural member", "spreading mycelium", "moisture > 20%"],
        "affected_substrates": ["wood"]
    },
    "warping_wood": {
        "id": "warping_wood",
        "name": "Warping (Wood / Parchment)",
        "icomos_classification": "C.5.6 Deformation — warping",
        "aliases": ["cupping", "bowing", "dimensional distortion"],
        "description": "Anisotropic shrinkage-swelling from RH cycling causes warping in wood panels and parchment leaves.",
        "detection_method": "Flatness measurement; RH history correlation",
        "axum_detection": "humidity_damage proxy on organic substrates",
        "urgency_factors": ["paint layer cracking from warp", "binding stress", "extreme RH excursion"],
        "affected_substrates": ["wood", "parchment_vellum", "ivory_bone"]
    },
    "efflorescence_paint": {
        "id": "efflorescence_paint",
        "name": "Efflorescence Behind Paint",
        "icomos_classification": "C.5.1 Crystallization (paint layer context)",
        "aliases": ["salt behind paint", "paint blistering from salts"],
        "description": "Salts crystallize at the paint-ground interface, pushing off the pigment layer in blisters. Common on lime plaster murals in damp churches.",
        "detection_method": "Blister analysis; salt identification in blister fluid; moisture behind paint",
        "axum_detection": "paint_detachment + salt_risk on painted_surface_fresco",
        "urgency_factors": ["original mural pigment", "active blistering", "church damp environment"],
        "affected_substrates": ["painted_surface_fresco", "stucco", "plaster_lime"]
    },
    "humidity_damage": {
        "id": "humidity_damage",
        "name": "Humidity Damage",
        "icomos_classification": "C.5.6 Deformation — humidity-related",
        "aliases": ["RH damage", "moisture damage", "swelling-shrinkage"],
        "description": "Cyclic or sustained incorrect RH causes dimensional change, adhesive failure, and biological growth on organic and composite materials.",
        "detection_method": "RH logger data; dimensional measurement; mould survey",
        "axum_detection": "active_moisture on organic substrates",
        "urgency_factors": ["RH outside 45–55%", "valuable organic artefact", "no climate control"],
        "affected_substrates": ["parchment_vellum", "wood", "paper", "textile_linen", "leather", "ivory_bone", "gilded_surface", "glass", "gypsum_alabaster"]
    },
    "accretion_deposits": {
        "id": "accretion_deposits",
        "name": "Accretion / Deposits",
        "icomos_classification": "D.1 Deposition — accretions",
        "aliases": ["calcareous deposit", "mineral accretion", "concretion"],
        "description": "Hard mineral deposits (calcite, silica) from groundwater runoff, bird droppings, or cementitious repair leachate.",
        "detection_method": "Hardness test; acid drop (calcite fizzes); cross-section",
        "axum_detection": "surface hardness anomaly; visual encrustation",
        "urgency_factors": ["obscuring detail", "trapped moisture beneath", "acid-generating deposit"],
        "affected_substrates": ["limestone_porous", "basalt", "glass", "metal_bronze", "terracotta_ceramic"]
    },
    "alveolization": {
        "id": "alveolization",
        "name": "Alveolization",
        "icomos_classification": "C.3 Erosion — alveolization",
        "aliases": ["honeycomb weathering", "cavernous weathering", "tafoni"],
        "description": "Development of small cavities (alveoli) on exposed stone faces, especially sandstone and basalt. Caused by salt crystallization, wind, and differential cement dissolution.",
        "detection_method": "Cavity depth mapping; surface roughness index",
        "axum_detection": "photometric stereo cavity detection on exposed stone",
        "urgency_factors": ["deep cavities", "structural thinning", "sculptural surface"],
        "affected_substrates": ["sandstone", "basalt", "tuff_ignimbrite", "travertine", "conglomerate", "laterite"]
    },
    "honeycombing": {
        "id": "honeycombing",
        "name": "Honeycombing",
        "icomos_classification": "C.3 Erosion — honeycombing",
        "aliases": ["severe alveolization", "network cavity weathering"],
        "description": "Advanced alveolization where cavities interconnect forming a honeycomb network. Indicates severe long-term weathering.",
        "detection_method": "Advanced cavity survey; ultrasonic thickness gauging",
        "axum_detection": "severe alveolization pattern on basalt/sandstone",
        "urgency_factors": ["wall thickness reduced", "structural concern", "monumental sculpture"],
        "affected_substrates": ["basalt", "sandstone", "tuff_ignimbrite", "travertine", "conglomerate"]
    },
    "blistering": {
        "id": "blistering",
        "name": "Blistering",
        "icomos_classification": "A.2 Surface damage — blistering",
        "aliases": ["surface blister", "paint blister", "render blister"],
        "description": "Localized dome-shaped detachments caused by vapour pressure, salt crystallization, or trapped moisture behind impermeable coatings.",
        "detection_method": "Sounding (hollow sound); cross-section of blister; moisture behind",
        "axum_detection": "stress_score + salt_risk on coated/plastered surfaces",
        "urgency_factors": ["impermeable coating present", "large blisters", "original surface beneath"],
        "affected_substrates": ["painted_surface_fresco", "plaster_lime", "stucco", "gilded_surface", "terracotta_ceramic"]
    },
    "rising_damp": {
        "id": "rising_damp",
        "name": "Rising Damp",
        "icomos_classification": "C.5.4 Moisture — rising damp",
        "aliases": ["groundwater rise", "damp proof course failure"],
        "description": "Persistent upward moisture movement through capillary networks from ground contact. Tide marks and salt accumulation at evaporation line.",
        "detection_method": "Gravimetric moisture profile; calcium carbide meter; salt at tide mark",
        "axum_detection": "active_moisture at base of earthen/volcanic structures",
        "urgency_factors": ["no DPC", "tide mark > 1m", "mural at damp zone"],
        "affected_substrates": ["earthen_mud_brick", "tuff_ignimbrite", "limestone_porous", "plaster_lime", "laterite", "sandstone"]
    },
    "soiling": {
        "id": "soiling",
        "name": "Soiling",
        "icomos_classification": "D.1 Deposition — soiling",
        "aliases": ["surface dirt", "dust accumulation", "particulate deposition"],
        "description": "Accumulation of dust, soot, and particulates on surfaces. Can be acidic, hold moisture, and obscure decorative or inscribed detail.",
        "detection_method": "Colour measurement (darkening); gravimetric dust collection; SEM particulate analysis",
        "axum_detection": "multispectral surface darkening; OCR obscuration",
        "urgency_factors": ["acidic soot", "biological growth on soiling", "inscription obscured"],
        "affected_substrates": ["limestone_porous", "marble", "basalt", "sandstone", "metal_bronze", "painted_surface_fresco", "textile_linen", "gilded_surface", "glass", "parchment_vellum"]
    },
    "patina_stable": {
        "id": "patina_stable",
        "name": "Stable Patina",
        "icomos_classification": "C.5.2 Metal corrosion — stable patina (protective)",
        "aliases": ["protective patina", "passive layer", "noble patina"],
        "description": "Well-formed corrosion layer (e.g. cuprite on bronze) that passivates the metal and protects from further attack. Should be preserved, not removed.",
        "detection_method": "Visual uniformity; no powdering; chloride test negative; stable for >10 years",
        "axum_detection": "metal_bronze without bronze_disease indicators",
        "urgency_factors": ["do not clean aggressively", "document colour and texture", "maintain stable RH"],
        "affected_substrates": ["metal_bronze", "metal_iron_steel"]
    },
    "patina_unstable": {
        "id": "patina_unstable",
        "name": "Unstable Patina",
        "icomos_classification": "C.5.2 Metal corrosion — unstable patina",
        "aliases": ["active patina", "powdery patina", "pustular corrosion"],
        "description": "Non-protective, actively growing corrosion layer that will continue to consume metal. Distinguished from stable patina by powdering and cyclic reformation.",
        "detection_method": "Powder test (q-tip); cyclic humidity test; chloride analysis",
        "axum_detection": "bronze_disease indicators; powdery surface on metal",
        "urgency_factors": ["powdering", "chloride present", "post-excavation"],
        "affected_substrates": ["metal_bronze", "metal_iron_steel"]
    },
}

# Fix efflorescence affected_substrates - remove invalid "brick" reference
DECAY_PATTERNS["efflorescence"]["affected_substrates"] = [
    s for s in DECAY_PATTERNS["efflorescence"]["affected_substrates"] if s != "brick"
]
DECAY_PATTERNS["capillary_rise_damage"]["affected_substrates"] = [
    s for s in DECAY_PATTERNS["capillary_rise_damage"]["affected_substrates"] if s != "brick"
]

TREATMENTS = {
    "desalination_poultice": {
        "id": "desalination_poultice",
        "name": "Desalination Poultice (Cellulose/Laponite RD)",
        "type": "DESALINATION",
        "compatible_substrates": ["limestone_porous", "sandstone", "tuff_ignimbrite", "terracotta_ceramic", "plaster_lime", "basalt", "dolomite", "travertine", "earthen_mud_brick"],
        "compatible_decay": ["salt_crystallization", "subflorescence", "efflorescence", "capillary_rise_damage", "rising_damp", "contour_scaling", "efflorescence_paint"],
        "never_use_on": ["metal_bronze", "metal_iron_steel", "parchment_vellum", "paper", "gilded_surface"],
        "application": "Apply 2–3 cm wet poultice (cellulose pulp + deionised water, or Laponite RD) to dry surface. Cover with polyethylene. Remove after 24–48 h. Repeat until conductivity < 300 µS/cm. Document each cycle.",
        "source": "Doehne & Price (2010); EN 16085:2012; GCI Salt Weathering Manual",
        "notes": "Surface must be dry before application. Do not poultice below 5°C."
    },
    "ethanol_water_cleaning": {
        "id": "ethanol_water_cleaning",
        "name": "Ethanol-Water Solution Cleaning (50:50)",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "basalt", "sandstone", "metal_bronze", "painted_surface_fresco"],
        "compatible_decay": ["soiling", "biological_colonization", "biological_crust_inscription", "accretion_deposits"],
        "never_use_on": ["parchment_vellum", "paper", "textile_linen", "gilded_surface"],
        "application": "Apply 50:50 ethanol:deionised water with swab on small test area first. Work in 10×10 cm sections. Blot dry. Repeat until swab is clean.",
        "source": "Fidler (2005); EN 16085:2012",
        "notes": "Test first on pigmented areas. Adequate ventilation required."
    },
    "dry_brush_cleaning": {
        "id": "dry_brush_cleaning",
        "name": "Dry Brush / Vacuum Cleaning",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "basalt", "sandstone", "parchment_vellum", "painted_surface_fresco", "wood", "textile_linen", "gilded_surface"],
        "compatible_decay": ["soiling", "biological_colonization", "mould_biological_organic"],
        "never_use_on": [],
        "application": "Soft natural-bristle brush (sable or hog) with HEPA vacuum capture. Brush toward vacuum nozzle. No water. Document before/after.",
        "source": "ICOMOS-ISCS 2008; Collections Care guidelines",
        "notes": "First-line cleaning for fragile surfaces. Never brush friable powdering stone aggressively."
    },
    "paraloid_b72_consolidant": {
        "id": "paraloid_b72_consolidant",
        "name": "Paraloid B-72 Consolidant (10% in Acetone/Ethanol)",
        "type": "CONSOLIDANT",
        "compatible_substrates": ["limestone_porous", "sandstone", "tuff_ignimbrite", "terracotta_ceramic", "plaster_lime"],
        "compatible_decay": ["granular_disintegration", "contour_scaling", "spalling", "sanding", "flaking"],
        "never_use_on": ["metal_bronze", "parchment_vellum", "paper", "gilded_surface", "painted_surface_fresco"],
        "incompatible_with_decay": ["salt_crystallization", "subflorescence", "efflorescence"],
        "incompatibility_reason_decay": "Consolidant traps soluble salts in stone pores, accelerating subflorescence spalling. Desalinate first.",
        "application": "Apply 10% w/v Paraloid B-72 in acetone/ethanol 1:1 by brush or pipette after desalination complete. 2–3 applications until rejection. Cap with 5% solution.",
        "source": "GCI Paraloid B-72 Technical Bulletin; Fidler (2005)",
        "notes": "Only after salt equilibrium achieved. Test absorption first."
    },
    "ethyl_silicate_consolidant": {
        "id": "ethyl_silicate_consolidant",
        "name": "Ethyl Silicate Consolidant (TEOS-based)",
        "type": "CONSOLIDANT",
        "compatible_substrates": ["sandstone", "tuff_ignimbrite", "conglomerate", "basalt"],
        "compatible_decay": ["granular_disintegration", "alveolization", "honeycombing", "sanding"],
        "never_use_on": ["limestone_porous", "marble", "painted_surface_fresco", "metal_bronze", "parchment_vellum"],
        "incompatible_substrates": ["limestone_porous", "marble", "dolomite", "travertine"],
        "incompatibility_reason": "Alkoxysilanes do not bond effectively to carbonate matrices and can leave white silica residues on limestone.",
        "application": "Apply KSE 300 or similar TEOS-based consolidant in 2–3 passes with 24 h intervals. Pre-wet with ethanol if surface is very dry.",
        "source": "Alkoxysilanes State of the Art Review (2015); EN 16085:2012",
        "notes": "For siliceous stone only. Not for carbonate."
    },
    "lime_water_consolidation": {
        "id": "lime_water_consolidation",
        "name": "Limewater Consolidation (Calcium Hydroxide)",
        "type": "CONSOLIDANT",
        "compatible_substrates": ["limestone_porous", "limestone_dense", "marble", "plaster_lime", "dolomite", "painted_surface_fresco"],
        "compatible_decay": ["granular_disintegration", "dissolution", "sugaring", "contour_scaling"],
        "never_use_on": ["sandstone", "metal_bronze", "wood", "parchment_vellum"],
        "incompatible_with_decay": ["salt_crystallization", "subflorescence"],
        "incompatibility_reason_decay": "Limewater adds calcium ions that can form insoluble salts with sulfate/chloride contaminants.",
        "application": "Apply saturated limewater (Ca(OH)2) by fine mist spray. Repeat weekly for 6–12 months. Carbonation produces calcite cement.",
        "source": "Doehne & Price (2010); ICOMOS Principles 2003",
        "notes": "Very slow but compatible with carbonate heritage. Requires patience."
    },
    "thymol_vapour_biocide": {
        "id": "thymol_vapour_biocide",
        "name": "Thymol Vapour Biocide Treatment",
        "type": "BIOCIDE",
        "compatible_substrates": ["parchment_vellum", "paper", "wood", "textile_linen", "leather"],
        "compatible_decay": ["mould_biological_organic", "insect_infestation"],
        "never_use_on": ["metal_bronze", "painted_surface_fresco", "limestone_porous"],
        "application": "Enclose artefact in sealed chamber with thymol crystals (25–50 g/m³) at 25°C for 3–4 weeks. Aerate 48 h before handling. Use fume hood.",
        "source": "ICON Mould Remediation Guidelines; AIC Paper Conservation Catalog",
        "notes": "Toxic vapour — PPE and ventilation mandatory. Not for open stone surfaces."
    },
    "bta_benzotriazole_inhibitor": {
        "id": "bta_benzotriazole_inhibitor",
        "name": "BTA (Benzotriazole) Corrosion Inhibitor",
        "type": "INHIBITOR",
        "compatible_substrates": ["metal_bronze"],
        "compatible_decay": ["bronze_disease", "patina_unstable", "pitting_corrosion"],
        "never_use_on": ["limestone_porous", "parchment_vellum", "painted_surface_fresco"],
        "application": "Apply 3% BTA in ethanol by brush after chloride-reduction cleaning. Repeat 3 times. Seal with microcrystalline wax if RH control unavailable.",
        "source": "CCI Notes 9/3; Scott (2002) Copper and Bronze Conservation",
        "notes": "BTA is a suspected carcinogen — use gloves and avoid inhalation."
    },
    "oxalic_acid_poultice_iron": {
        "id": "oxalic_acid_poultice_iron",
        "name": "Oxalic Acid Poultice (Iron Stain Removal)",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "sandstone", "plaster_lime"],
        "compatible_decay": ["iron_staining"],
        "never_use_on": ["metal_bronze", "metal_iron_steel", "parchment_vellum", "painted_surface_fresco", "gilded_surface"],
        "incompatible_with_decay": ["biological_colonization"],
        "incompatibility_reason_decay": "Acid kills biota but damages carbonate; only for iron stain on sound stone.",
        "application": "10% oxalic acid in cellulose poultice, apply 2 h max. Rinse thoroughly with deionised water. Neutralise with dilute ammonia. Test first.",
        "source": "Doehne & Price (2010); Ashurst (1998)",
        "notes": "Acid treatment — can etch carbonate. Limited contact time essential."
    },
    "calcium_phytate_treatment": {
        "id": "calcium_phytate_treatment",
        "name": "Calcium Phytate Treatment (Iron Gall Ink Stabilisation)",
        "type": "STABILISATION",
        "compatible_substrates": ["parchment_vellum", "paper"],
        "compatible_decay": ["ink_corrosion"],
        "never_use_on": ["limestone_porous", "metal_bronze", "painted_surface_fresco"],
        "application": "Apply 2% calcium phytate solution by brush to ink lines, followed by 0.5% calcium bicarbonate deacidification. Reduces iron-catalysed hydrolysis.",
        "source": "Banik et al. (2009); EU Paper Conservation research",
        "notes": "For iron gall ink manuscripts only. Test on marginalia first."
    },
    "wheat_starch_paste_consolidant": {
        "id": "wheat_starch_paste_consolidant",
        "name": "Wheat Starch Paste (Paper/Parchment Repair)",
        "type": "CONSOLIDANT",
        "compatible_substrates": ["parchment_vellum", "paper", "textile_linen"],
        "compatible_decay": ["delamination_parchment", "crack_mechanical", "flaking"],
        "never_use_on": ["limestone_porous", "metal_bronze", "basalt", "painted_surface_fresco"],
        "application": "Thin wheat starch paste (Japanese paper method) for tear repair and lining. Apply with bamboo spatula. Press between blotters 24 h.",
        "source": "AIC Paper Conservation Catalog; Etherington & Roberts",
        "notes": "Reversible repair adhesive. Not for stone."
    },
    "controlled_rh_humidification": {
        "id": "controlled_rh_humidification",
        "name": "Controlled RH Humidification Chamber",
        "type": "ENVIRONMENTAL",
        "compatible_substrates": ["parchment_vellum", "wood", "paper", "textile_linen", "leather", "ivory_bone"],
        "compatible_decay": ["humidity_damage", "warping_wood", "delamination_parchment", "crack_mechanical"],
        "never_use_on": ["metal_bronze"],
        "application": "Place artefact in chamber at target RH 50% (±3%). Increase 2% per day from current. Hold 2 weeks. Reduce 2% per day. Monitor with data logger.",
        "source": "ASHRAE Chapter 24; Michalski (2002) Climate Guidelines",
        "notes": "For flattening warped parchment and relaxing brittle organic materials."
    },
    "hot_lime_mortar_repair": {
        "id": "hot_lime_mortar_repair",
        "name": "Hot Lime Mortar Repair (Calcium Lime)",
        "type": "REPAIR",
        "compatible_substrates": ["limestone_porous", "basalt", "sandstone", "plaster_lime", "earthen_mud_brick", "tuff_ignimbrite"],
        "compatible_decay": ["crack_mechanical", "spalling", "previous_treatment_failure", "rising_damp"],
        "never_use_on": ["metal_bronze", "parchment_vellum", "gilded_surface", "painted_surface_fresco"],
        "application": "Prepare hot mixed lime putty mortar (1:3 lime:aggregate matching original). Pack into void after removing failed repair. Cure 28 days under damp hessian.",
        "source": "ICOMOS Principles 2003; EN 16085:2012; Historic England Lime Guidance",
        "notes": "Compatible permeability with historic masonry. Never use Portland cement."
    },
    "nhl_mortar_repair": {
        "id": "nhl_mortar_repair",
        "name": "NHL Natural Hydraulic Lime Mortar Repair",
        "type": "REPAIR",
        "compatible_substrates": ["limestone_porous", "basalt", "sandstone", "plaster_lime", "earthen_mud_brick"],
        "compatible_decay": ["crack_mechanical", "spalling", "contour_scaling", "previous_treatment_failure"],
        "never_use_on": ["metal_bronze", "parchment_vellum", "gilded_surface"],
        "application": "NHL 3.5 or NHL 5 mortar matched to substrate hardness. Dampen substrate. Apply in layers. Protect from rapid drying 7 days.",
        "source": "EN 459-1; ICOMOS Principles 2003",
        "notes": "NHL 2 for soft stone, NHL 5 for hard basalt/granite."
    },
    "microcrystalline_wax_coating": {
        "id": "microcrystalline_wax_coating",
        "name": "Microcrystalline Wax Coating (Renaissance Wax)",
        "type": "COATING",
        "compatible_substrates": ["metal_bronze", "metal_iron_steel"],
        "compatible_decay": ["bronze_disease", "rust_corrosion", "patina_unstable", "soiling"],
        "never_use_on": ["limestone_porous", "sandstone", "parchment_vellum", "painted_surface_fresco"],
        "application": "Apply thin film of microcrystalline wax with soft cloth after cleaning and inhibitor treatment. Buff to satin finish. Renew every 5–10 years.",
        "source": "CCI Notes 9/7; Plenderleith & Werner (1971)",
        "notes": "Only after active corrosion arrested. Not for porous stone (traps moisture)."
    },
    "poultice_cleaning": {
        "id": "poultice_cleaning",
        "name": "Clay Poultice Cleaning (Sevofix/Laponite)",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "sandstone", "basalt", "plaster_lime"],
        "compatible_decay": ["black_crust", "soiling", "accretion_deposits", "biological_colonization"],
        "never_use_on": ["parchment_vellum", "gilded_surface", "painted_surface_fresco"],
        "application": "Apply clay poultice (Laponite RD or bentonite) with deionised water or appropriate chelator. Cover 24 h. Remove gently. Rinse residue.",
        "source": "Doehne & Price (2010); EN 16085:2012",
        "notes": "For gypsum crust and surface deposits on stone."
    },
    "laser_cleaning": {
        "id": "laser_cleaning",
        "name": "Laser Cleaning (Nd:YAG 1064 nm)",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "sandstone", "basalt", "metal_bronze"],
        "compatible_decay": ["black_crust", "soiling", "biological_colonization", "accretion_deposits"],
        "never_use_on": ["parchment_vellum", "painted_surface_fresco", "gilded_surface"],
        "application": "Nd:YAG laser at 1064 nm, fluence 0.5–1.5 J/cm², 6–10 Hz. Test patch mandatory. Operator certified. Fume extraction required.",
        "source": "Cooper (1998) Laser Cleaning in Conservation; EN 16085:2012",
        "notes": "Professional equipment only. Risk of yellowing on white marble at high fluence."
    },
    "ammonium_carbonate_poultice": {
        "id": "ammonium_carbonate_poultice",
        "name": "Ammonium Carbonate Poultice (Gypsum Conversion)",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "plaster_lime"],
        "compatible_decay": ["black_crust", "accretion_deposits"],
        "never_use_on": ["sandstone", "metal_bronze", "painted_surface_fresco", "gilded_surface"],
        "application": "5–10% ammonium carbonate in cellulose poultice. Converts gypsum crust to soluble ammonium sulfate. Rinse thoroughly. Max 4 h contact.",
        "source": "Ashurst (1998); ICOMOS-ISCS 2008",
        "notes": "Converts gypsum to soluble salt — must rinse completely or salt damage follows."
    },
    "edta_chelating_poultice": {
        "id": "edta_chelating_poultice",
        "name": "EDTA Chelating Poultice (Iron/Stain Removal)",
        "type": "CLEANING",
        "compatible_substrates": ["limestone_porous", "marble", "sandstone", "plaster_lime"],
        "compatible_decay": ["iron_staining", "black_crust"],
        "never_use_on": ["metal_bronze", "metal_iron_steel", "painted_surface_fresco", "gilded_surface"],
        "application": "3% EDTA disodium salt in cellulose poultice, pH 7–8. Apply 2–4 h. Rinse with deionised water. Follow with desalination if sulfate residues.",
        "source": "Doehne & Price (2010); Torraca (1984)",
        "notes": "Chelates iron and calcium — can etch carbonate if left too long."
    },
    "hydrogen_peroxide_bleaching": {
        "id": "hydrogen_peroxide_bleaching",
        "name": "Hydrogen Peroxide Bleaching (Controlled)",
        "type": "CLEANING",
        "compatible_substrates": ["parchment_vellum", "paper", "textile_linen"],
        "compatible_decay": ["mould_biological_organic", "iron_staining"],
        "never_use_on": ["limestone_porous", "metal_bronze", "painted_surface_fresco", "gilded_surface", "basalt"],
        "application": "1–3% H2O2 applied locally to stained organic material under magnification. Neutralise with water. Avoid cellulose degradation at high concentration.",
        "source": "AIC Paper Conservation Catalog",
        "notes": "Can weaken cellulose fibres. Test first. Not for coloured pigments."
    },
    "silicone_sealant": {
        "id": "silicone_sealant",
        "name": "Silicone Sealant / Water Repellent",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "tuff_ignimbrite", "marble", "plaster_lime", "painted_surface_fresco", "terracotta_ceramic", "earthen_mud_brick"],
        "danger_level": "critical",
        "danger_reason": "Silicone forms impermeable film trapping moisture and salts behind surface, accelerating subflorescence spalling. Irreversible. Rejects water vapour egress causing blistering.",
        "source": "Fidler (2005) — 60% practitioner error rate; ICOMOS Principles 2003; Ashurst (1998)",
        "notes": "NEVER apply to historic porous masonry."
    },
    "portland_cement_patching": {
        "id": "portland_cement_patching",
        "name": "Portland Cement Patching / Repair",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "tuff_ignimbrite", "marble", "plaster_lime", "earthen_mud_brick", "terracotta_ceramic"],
        "danger_level": "critical",
        "danger_reason": "Portland cement is harder, impermeable, and thermally incompatible with historic lime-based masonry. Creates stress concentrations at interface causing spalling of original fabric. Traps moisture.",
        "source": "ICOMOS Principles 2003; Historic England; Fidler (2005)",
        "notes": "Use NHL or hot lime mortar instead."
    },
    "acrylic_sealant": {
        "id": "acrylic_sealant",
        "name": "Acrylic Sealant / Surface Coating",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "tuff_ignimbrite", "marble", "plaster_lime", "painted_surface_fresco"],
        "danger_level": "critical",
        "danger_reason": "Acrylic films are impermeable to water vapour, trapping moisture and salts. Yellow and cross-link irreversibly. Peel taking original surface with them.",
        "source": "Fidler (2005); GCI Coatings Review; ICOMOS Principles 2003",
        "notes": "AXUM synthetic test expects this flagged on limestone inscription cases."
    },
    "pressure_washing": {
        "id": "pressure_washing",
        "name": "Pressure Washing",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "tuff_ignimbrite", "marble", "plaster_lime", "painted_surface_fresco", "parchment_vellum", "terracotta_ceramic"],
        "danger_level": "critical",
        "danger_reason": "High-pressure water drives moisture and salts deeper into porous stone, erodes inscription detail, and damages friable surfaces irreversibly.",
        "source": "Fidler (2005); ICOMOS-ISCS 2008",
        "notes": "Common field error — never use on heritage stone."
    },
    "hydrofluoric_acid_cleaning": {
        "id": "hydrofluoric_acid_cleaning",
        "name": "Hydrofluoric Acid Cleaning",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "marble", "sandstone", "basalt", "metal_bronze", "painted_surface_fresco", "parchment_vellum"],
        "danger_level": "critical",
        "danger_reason": "HF dissolves silicate minerals and attacks glass, ceramics, and stone matrix. Extremely toxic. Causes irreversible etching and loss of surface detail.",
        "source": "Health and Safety Executive; Conservation ethics",
        "notes": "Banned from conservation practice."
    },
    "epoxy_resin_consolidant": {
        "id": "epoxy_resin_consolidant",
        "name": "Epoxy Resin Consolidant",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "parchment_vellum", "painted_surface_fresco", "wood", "terracotta_ceramic"],
        "danger_level": "high",
        "danger_reason": "Epoxy is irreversible, discolours, creates impermeable barrier, and has mismatched thermal expansion causing internal stress cracking.",
        "source": "Fidler (2005); GCI Consolidants Review",
        "notes": "Use Paraloid B-72 or ethyl silicate instead."
    },
    "bleach_sodium_hypochlorite": {
        "id": "bleach_sodium_hypochlorite",
        "name": "Sodium Hypochlorite Bleach (Household Bleach)",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "metal_bronze", "parchment_vellum", "paper", "textile_linen", "painted_surface_fresco", "wood"],
        "danger_level": "critical",
        "danger_reason": "NaOCl oxidises organic materials, corrodes metals, deposits sodium chloride crystals in stone pores, and bleaches pigments irreversibly.",
        "source": "Fidler (2005); AIC safety guidelines",
        "notes": "Common household error on mouldy manuscripts and stone."
    },
    "mechanical_grinding": {
        "id": "mechanical_grinding",
        "name": "Mechanical Grinding / Sandblasting",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "marble", "painted_surface_fresco", "metal_bronze", "parchment_vellum"],
        "danger_level": "critical",
        "danger_reason": "Abrasive removal destroys original surface tool marks, inscriptions, and patina. Opens fresh pore surfaces accelerating decay.",
        "source": "ICOMOS-ISCS 2008; EN 16085:2012",
        "notes": "Destroys archaeological surface evidence."
    },
    "biocide_sodium_hypochlorite_stone": {
        "id": "biocide_sodium_hypochlorite_stone",
        "name": "Sodium Hypochlorite Biocide on Stone",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "marble", "sandstone", "basalt", "plaster_lime", "painted_surface_fresco"],
        "danger_level": "high",
        "danger_reason": "Chlorine residues deposit chloride ions in stone pores, priming severe salt crystallization and subflorescence after treatment.",
        "source": "Doehne & Price (2010); ICOMOS-ISCS 2008",
        "notes": "Use dry brush + ethanol or approved biocide instead."
    },
    "hard_cement_grout": {
        "id": "hard_cement_grout",
        "name": "Hard Cement Grout Injection",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "tuff_ignimbrite", "earthen_mud_brick", "mosaic_tesserae"],
        "danger_level": "critical",
        "danger_reason": "Rigid cement grout prevents thermal and moisture movement, concentrates stress at stone interface, and causes adjacent stone to fracture.",
        "source": "ICOMOS Principles 2003; Fidler (2005)",
        "notes": "Use lime-based grout matched to substrate."
    },
    "parchment_lamination_pva": {
        "id": "parchment_lamination_pva",
        "name": "PVA Lamination of Manuscripts",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["parchment_vellum", "paper"],
        "danger_level": "critical",
        "danger_reason": "PVA is irreversible, stiffens collagen, accelerates embrittlement, and cannot be removed without destroying the manuscript.",
        "source": "AIC Paper Conservation Catalog; Fidler (2005)",
        "notes": "Use wheat starch paste and Japanese tissue instead."
    },
    "wd40_metal_treatment": {
        "id": "wd40_metal_treatment",
        "name": "WD-40 / Oil Penetrant on Bronze",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["metal_bronze", "metal_iron_steel"],
        "danger_level": "high",
        "danger_reason": "Petroleum oils attract dust, yellow over time, contain solvents that strip stable patina, and do not inhibit bronze disease chlorides.",
        "source": "CCI Notes 9/3; Scott (2002)",
        "notes": "Use BTA inhibitor and microcrystalline wax instead."
    },
    "consolidant_injection_epoxy_stone": {
        "id": "consolidant_injection_epoxy_stone",
        "name": "Epoxy Injection into Stone Cracks",
        "type": "DANGEROUS_TREATMENT",
        "compatible_substrates": [],
        "compatible_decay": [],
        "never_use_on": ["limestone_porous", "sandstone", "basalt", "marble", "tuff_ignimbrite"],
        "danger_level": "high",
        "danger_reason": "Epoxy in cracks prevents moisture egress, has mismatched elasticity, and is irreversible. Cracks re-open adjacent to epoxy fill.",
        "source": "GCI Consolidants Review; ICOMOS Principles 2003",
        "notes": "Use lime grout or stainless steel pinning with lime mortar."
    },
}

EXPLICIT_MATRIX = {
    "limestone_porous": {
        "salt_crystallization": {
            "safe": ["desalination_poultice", "dry_brush_cleaning", "poultice_cleaning"],
            "unsafe": [
                {"treatment": "paraloid_b72_consolidant", "risk": "high", "mechanism": "Traps soluble salts causing accelerated subflorescence spalling", "source": "GCI Salt Weathering Manual"},
                {"treatment": "acrylic_sealant", "risk": "critical", "mechanism": "Impermeable film traps moisture and salts behind surface", "source": "Fidler (2005)"},
                {"treatment": "silicone_sealant", "risk": "critical", "mechanism": "Blocks vapour transmission causing blistering and salt accumulation", "source": "ICOMOS Principles 2003"},
                {"treatment": "portland_cement_patching", "risk": "critical", "mechanism": "Hard incompatible repair causes stress spalling at interface", "source": "Historic England"},
            ],
            "sequence": "Document → isolate moisture → desalination poultice cycles (conductivity < 300 µS/cm) → monitor 4 weeks → consolidant only if powdering persists",
            "safe_after_desalination": ["paraloid_b72_consolidant", "lime_water_consolidation"],
        },
        "biological_crust_inscription": {
            "safe": ["dry_brush_cleaning", "ethanol_water_cleaning", "poultice_cleaning"],
            "unsafe": [
                {"treatment": "pressure_washing", "risk": "critical", "mechanism": "Erodes inscription grooves and drives moisture into stone", "source": "ICOMOS-ISCS 2008"},
                {"treatment": "biocide_sodium_hypochlorite_stone", "risk": "high", "mechanism": "Deposits chloride ions priming salt crystallization", "source": "Doehne & Price (2010)"},
                {"treatment": "mechanical_grinding", "risk": "critical", "mechanism": "Removes inscription detail irreversibly", "source": "EN 16085:2012"},
            ],
            "sequence": "Document inscription (photogrammetry) → dry brush → local ethanol-water swab → assess legibility → biocide only if growth reoccurs",
        },
        "granular_disintegration": {
            "safe": ["desalination_poultice", "lime_water_consolidation"],
            "unsafe": [
                {"treatment": "acrylic_sealant", "risk": "critical", "mechanism": "Seals powdering surface trapping hygroscopic salts", "source": "Fidler (2005)"},
                {"treatment": "pressure_washing", "risk": "critical", "mechanism": "Removes remaining sound grain structure", "source": "ICOMOS-ISCS 2008"},
            ],
            "sequence": "Desalinate first if salts present → limewater consolidation (6–12 months) or Paraloid B-72 after salt equilibrium",
            "safe_after_desalination": ["paraloid_b72_consolidant", "lime_water_consolidation"],
        },
    },
    "marble": {
        "sugaring": {
            "safe": ["dry_brush_cleaning", "lime_water_consolidation", "poultice_cleaning"],
            "unsafe": [
                {"treatment": "oxalic_acid_poultice_iron", "risk": "high", "mechanism": "Acid dissolves calcite crystals accelerating sugaring", "source": "Doehne & Price (2010)"},
                {"treatment": "pressure_washing", "risk": "critical", "mechanism": "Mechanically removes crystal grains", "source": "ICOMOS-ISCS 2008"},
            ],
            "sequence": "Gentle dry cleaning → limewater consolidation → environmental shelter from rain",
        },
        "black_crust": {
            "safe": ["poultice_cleaning", "ammonium_carbonate_poultice", "laser_cleaning"],
            "unsafe": [
                {"treatment": "hydrofluoric_acid_cleaning", "risk": "critical", "mechanism": "Attacks calcite matrix", "source": "Conservation ethics"},
                {"treatment": "mechanical_grinding", "risk": "critical", "mechanism": "Removes original polished surface", "source": "EN 16085:2012"},
            ],
            "sequence": "Test patch → ammonium carbonate poultice OR laser → rinse → limewater aftercare",
        },
    },
    "sandstone": {
        "alveolization": {
            "safe": ["desalination_poultice", "ethyl_silicate_consolidant", "dry_brush_cleaning"],
            "unsafe": [
                {"treatment": "acrylic_sealant", "risk": "critical", "mechanism": "Traps salts in cavity zones", "source": "Fidler (2005)"},
                {"treatment": "portland_cement_patching", "risk": "critical", "mechanism": "Hard fill causes differential erosion around repair", "source": "ICOMOS Principles 2003"},
            ],
            "sequence": "Desalinate → ethyl silicate consolidant (2–3 passes) → monitor erosion rate",
            "safe_after_desalination": ["ethyl_silicate_consolidant"],
        },
        "salt_crystallization": {
            "safe": ["desalination_poultice", "poultice_cleaning"],
            "unsafe": [
                {"treatment": "paraloid_b72_consolidant", "risk": "high", "mechanism": "Salt trapped by consolidant causes contour scaling", "source": "GCI"},
            ],
            "sequence": "Desalination cycles until equilibrium → consolidant if friable",
            "safe_after_desalination": ["ethyl_silicate_consolidant", "paraloid_b72_consolidant"],
        },
    },
    "basalt": {
        "spalling": {
            "safe": ["dry_brush_cleaning", "nhl_mortar_repair", "hot_lime_mortar_repair"],
            "unsafe": [
                {"treatment": "portland_cement_patching", "risk": "critical", "mechanism": "Thermal mismatch causes new spalling adjacent to repair", "source": "NGU Bulletin 436"},
                {"treatment": "epoxy_resin_consolidant", "risk": "high", "mechanism": "Irreversible darkening and vapour barrier", "source": "GCI"},
            ],
            "sequence": "Remove loose spall → desalinate if salt present → NHL 5 mortar repair → cure under hessian",
        },
        "biological_colonization": {
            "safe": ["dry_brush_cleaning", "ethanol_water_cleaning"],
            "unsafe": [
                {"treatment": "pressure_washing", "risk": "critical", "mechanism": "Opens vesicles and drives water into basalt", "source": "ICOMOS-ISCS 2008"},
            ],
            "sequence": "Dry brush → local ethanol-water → monitor regrowth annually",
        },
    },
    "tuff_ignimbrite": {
        "capillary_rise_damage": {
            "safe": ["desalination_poultice", "hot_lime_mortar_repair", "dry_brush_cleaning"],
            "unsafe": [
                {"treatment": "silicone_sealant", "risk": "critical", "mechanism": "Seals damp zone trapping salts at tide mark", "source": "ICOMOS Principles 2003"},
                {"treatment": "portland_cement_patching", "risk": "critical", "mechanism": "Impermeable base repair accelerates rising damp above joint", "source": "ISCEAH 2019"},
            ],
            "sequence": "Install DPC if possible → desalinate tide mark zone → breathable lime render → monitor moisture profile",
        },
        "salt_crystallization": {
            "safe": ["desalination_poultice"],
            "unsafe": [
                {"treatment": "acrylic_sealant", "risk": "critical", "mechanism": "Highly porous tuff traps salts behind sealant", "source": "Fidler (2005)"},
            ],
            "sequence": "Extended desalination (10+ cycles typical for tuff) → consolidant only after conductivity stable",
            "safe_after_desalination": ["paraloid_b72_consolidant"],
        },
    },
    "terracotta_ceramic": {
        "salt_crystallization": {
            "safe": ["desalination_poultice", "dry_brush_cleaning"],
            "unsafe": [
                {"treatment": "pressure_washing", "risk": "critical", "mechanism": "Water ingress through cracks accelerates salt damage", "source": "Ceramics conservation guidelines"},
                {"treatment": "epoxy_resin_consolidant", "risk": "high", "mechanism": "Irreversible darkening and loss of surface", "source": "GCI"},
            ],
            "sequence": "Desalinate → assess joins → Paraloid B-72 on friable slip if needed",
            "safe_after_desalination": ["paraloid_b72_consolidant"],
        },
        "crack_mechanical": {
            "safe": ["wheat_starch_paste_consolidant"],
            "unsafe": [
                {"treatment": "portland_cement_patching", "risk": "critical", "mechanism": "Rigid fill causes crack propagation in fired body", "source": "ICOMOS Principles 2003"},
            ],
            "sequence": "Document cracks → adhesive repair with reversible adhesive → fill losses with compatible ceramic fill",
        },
    },
    "parchment_vellum": {
        "ink_corrosion": {
            "safe": ["calcium_phytate_treatment", "controlled_rh_humidification", "dry_brush_cleaning"],
            "unsafe": [
                {"treatment": "hydrogen_peroxide_bleaching", "risk": "high", "mechanism": "Oxidises ink and weakens collagen at perforation zones", "source": "Banik et al. (2009)"},
                {"treatment": "parchment_lamination_pva", "risk": "critical", "mechanism": "Irreversible stiffening accelerates ink burn-through", "source": "AIC"},
                {"treatment": "bleach_sodium_hypochlorite", "risk": "critical", "mechanism": "Destroys collagen and fades ink", "source": "Fidler (2005)"},
            ],
            "sequence": "RH stabilisation → calcium phytate on ink lines → deacidification → storage at 45–55% RH",
        },
        "delamination_parchment": {
            "safe": ["controlled_rh_humidification", "wheat_starch_paste_consolidant"],
            "unsafe": [
                {"treatment": "parchment_lamination_pva", "risk": "critical", "mechanism": "Traps moisture between layers causing further separation", "source": "AIC"},
            ],
            "sequence": "Humidify to 50% RH → align layers → starch paste repair → flatten under weight",
        },
        "mould_biological_organic": {
            "safe": ["dry_brush_cleaning", "thymol_vapour_biocide", "controlled_rh_humidification"],
            "unsafe": [
                {"treatment": "bleach_sodium_hypochlorite", "risk": "critical", "mechanism": "Irreversible collagen degradation", "source": "Fidler (2005)"},
                {"treatment": "ethanol_water_cleaning", "risk": "moderate", "mechanism": "Can cause tide lines and ink bleeding if over-applied", "source": "AIC"},
            ],
            "sequence": "Isolate → dry brush spores → thymol vapour → RH stabilise → monitor 6 months",
        },
    },
    "painted_surface_fresco": {
        "paint_detachment": {
            "safe": ["controlled_rh_humidification", "dry_brush_cleaning", "lime_water_consolidation"],
            "unsafe": [
                {"treatment": "acrylic_sealant", "risk": "critical", "mechanism": "Traps moisture behind paint causing further detachment", "source": "Fidler (2005)"},
                {"treatment": "pressure_washing", "risk": "critical", "mechanism": "Removes pigment layer", "source": "ICOMOS-ISCS 2008"},
                {"treatment": "epoxy_resin_consolidant", "risk": "high", "mechanism": "Darkens and stiffens pigment layer", "source": "GCI"},
            ],
            "sequence": "Document → RH stabilise → inject dilute lime water behind blisters → re-adhere flakes with reversible adhesive",
        },
        "efflorescence_paint": {
            "safe": ["desalination_poultice", "dry_brush_cleaning"],
            "unsafe": [
                {"treatment": "paraloid_b72_consolidant", "risk": "high", "mechanism": "Seals salts at paint-ground interface", "source": "GCI"},
            ],
            "sequence": "Desalinate behind paint layer via poultice → reassess adhesion → consolidate ground not pigment",
            "safe_after_desalination": ["lime_water_consolidation"],
        },
    },
    "metal_bronze": {
        "bronze_disease": {
            "safe": ["bta_benzotriazole_inhibitor", "microcrystalline_wax_coating", "dry_brush_cleaning"],
            "unsafe": [
                {"treatment": "wd40_metal_treatment", "risk": "high", "mechanism": "Does not remove chlorides; attracts dust; strips stable patina", "source": "CCI Notes 9/3"},
                {"treatment": "bleach_sodium_hypochlorite", "risk": "critical", "mechanism": "Deposits chlorides accelerating bronze disease cycle", "source": "Scott (2002)"},
                {"treatment": "hydrofluoric_acid_cleaning", "risk": "critical", "mechanism": "Dissolves metal surface", "source": "Conservation ethics"},
            ],
            "sequence": "Document → mechanical removal of powdery corrosion → chloride reduction → BTA 3% × 3 → wax coating → RH < 55%",
        },
        "patina_stable": {
            "safe": ["dry_brush_cleaning", "microcrystalline_wax_coating"],
            "unsafe": [
                {"treatment": "laser_cleaning", "risk": "high", "mechanism": "Removes protective patina exposing fresh metal to corrosion", "source": "Cooper (1998)"},
                {"treatment": "mechanical_grinding", "risk": "critical", "mechanism": "Destroys historic patina and surface evidence", "source": "CCI"},
            ],
            "sequence": "Do not clean aggressively → dust → wax if handling required → maintain stable RH",
        },
    },
}


def main() -> None:
    """Write all KB fragment JSON files."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    write("substrates.json", SUBSTRATES)
    write("decay_patterns.json", DECAY_PATTERNS)
    write("treatments.json", TREATMENTS)
    write("compatibility_matrix_explicit.json", EXPLICIT_MATRIX)
    print(f"Substrates: {len(SUBSTRATES)}")
    print(f"Decay patterns: {len(DECAY_PATTERNS)}")
    print(f"Treatments: {len(TREATMENTS)}")


if __name__ == "__main__":
    main()

