"""Call cells from sequencing reads using unified interface.

Supports both single-barcode and multi-barcode protocols.
"""

import pandas as pd

from lib.sbs.call_cells import call_cells, prep_multi_reads, call_cells_v2, call_cells_v3

# Get configuration from params
params = snakemake.params.config

# Load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load barcode library
df_barcode_library = pd.read_csv(params["df_barcode_library_fp"], sep="\t")

# Choose calling method based on barcode_type parameter
barcode_type = params.get("barcode_type", "simple")

if barcode_type == "multi":
    # Multi-barcode mode: prep reads first, then call cells
    df_reads = prep_multi_reads(
        reads_data,
        map_start=params["map_start"],
        map_end=params["map_end"],
        recomb_start=params["recomb_start"],
        recomb_end=params["recomb_end"],
        map_col=params["map_col"],
        recomb_col=params["recomb_col"],
    )

    cells_data = call_cells(
        reads_data=df_reads,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        map_start=params["map_start"],
        map_end=params["map_end"],
        map_col=params["map_col"],
        recomb_start=params["recomb_start"],
        recomb_end=params["recomb_end"],
        recomb_col=params["recomb_col"],
        recomb_filter_col=params["recomb_filter_col"],
        recomb_q_thresh=params["recomb_q_thresh"],
        error_correct=params["error_correct"],
        sort_calls=params["sort_calls"],
        max_distance=params["max_distance"],
        barcode_info_cols=params["barcode_info_cols"],
    )
elif barcode_type == "dual": 
    lib_lookup = {(row.prefix_map, row.prefix_recomb): row.gene_symbol for _, row in df_barcode_library.iterrows()}
    
    cells_data = call_cells_v3(
        df_reads_=reads_data,
        library_lookup=lib_lookup,
        alpha=params["alpha"],
        max_pair_dist=params["max_pair_dist"],
        max_assign_dist=params["max_assign_dist"],
        q_min=params["q_min"],
        max_q_mismatches=params["max_q_mismatches"],
        max_n_reads_in_cell=params["max_n_reads_in_cell"],
        max_n_unexplained_spots_in_cell=params["max_n_unexplained_spots_in_cell"],
        min_n_spots_in_cell=params["min_n_spots_in_cell"]
        )
else:
    # Simple barcode mode: call cells directly
    cells_data = call_cells(
        reads_data=reads_data,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        barcode_col=params["barcode_col"],
        prefix_col=params["prefix_col"],
        error_correct=params["error_correct"],
        sort_calls=params["sort_calls"],
        max_distance=params["max_distance"],
    )

# Save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
