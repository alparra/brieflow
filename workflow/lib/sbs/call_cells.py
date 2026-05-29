"""Unified utility functions for calling cells from sequencing reads.

This module supports both single-barcode and multi-barcode protocols with consistent function signatures.
Outputs are standardized across all modes for consistent downstream processing.
"""

import pandas as pd
import numpy as np
import Levenshtein

# constants for calling cells
from lib.sbs.constants import (
    PREFIX,
    SGRNA,
    GENE_SYMBOL,
    GENE_ID,
    WELL,
    TILE,
    CELL,
    READ,
    BARCODE,
    BARCODE_COUNT,
    BARCODE_0,
    BARCODE_1,
    BARCODE_COUNT_0,
    BARCODE_COUNT_1,
    POSITION_I,
    POSITION_J,
    UMI_0,
    UMI_1,
    UMI_COUNT,
    UMI_COUNT_0,
    UMI_COUNT_1,
)


def call_cells(
    reads_data,
    df_barcode_library=None,
    q_min=0,
    # Barcode extraction parameters (mutually exclusive modes)
    barcode_col="sgRNA",  # For auto-truncation mode
    prefix_col=None,  # For pre-computed prefix mode
    map_start=None,  # For cycle-based extraction (multi-barcode)
    map_end=None,
    map_col="prefix_map",  # Column name for mapping barcode
    # Recombination detection (multi-barcode only)
    recomb_start=None,
    recomb_end=None,
    recomb_col="prefix_recomb",  # Column name for recombination barcode
    recomb_filter_col=None,  # Quality column to filter recombination
    recomb_q_thresh=0.1,  # Threshold for recombination quality
    # Sorting and error correction
    sort_calls="peak",  # "peak" or "count"
    error_correct=False,
    max_distance=2,
    distance_metric="hamming",
    # Output customization
    barcode_info_cols=None,  # Defaults to [GENE_SYMBOL, GENE_ID] if available
    # Optional UMI support
    df_UMI=None,
    **kwargs,
):
    """Unified cell calling that supports both single and multi-barcode protocols.

    This function identifies cell barcodes from sequencing reads, with support for:
    - Single-barcode protocols (standard SBS)
    - Multi-barcode protocols (with recombination detection)
    - Count-based or peak-based sorting
    - Error correction with pre-correction tracking
    - UMI information (optional)

    BARCODE EXTRACTION MODES (mutually exclusive):

    1. **Cycle-based mode** (for multi-barcode):
       Specify map_start/map_end (and optionally recomb_start/recomb_end).
       Extracts specific cycle ranges from full barcode via slicing.
       Example: map_start=1, map_end=15, recomb_start=16, recomb_end=30

    2. **Pre-computed prefix mode**:
       Specify prefix_col with pre-computed prefixes in library.
       Useful for cycle-skipping scenarios.
       Example: prefix_col="custom_prefix"

    3. **Auto-truncation mode** (default):
       Uses barcode_col and automatically truncates to match read length.
       Example: barcode_col="sgRNA" (default)

    Args:
        reads_data (DataFrame): DataFrame containing read information with columns:
            well, tile, cell, read, barcode, peak, Q_min, Q_0, Q_1, ...
        df_barcode_library (DataFrame, optional): DataFrame containing barcode library.
            Must contain PREFIX column or map_col for mapping.
        q_min (int, optional): Minimum quality threshold. Default is 0.

        barcode_col (str, optional): Column in library with full sequences for
            auto-truncation. Default is 'sgRNA'.
        prefix_col (str, optional): Column in library with pre-computed prefixes.
            Overrides barcode_col if specified.
        map_start (int, optional): Starting cycle for mapping barcode (1-indexed).
        map_end (int, optional): Ending cycle for mapping barcode (1-indexed, inclusive).
        map_col (str, optional): Name for mapping barcode column. Default is 'prefix_map'.

        recomb_start (int, optional): Starting cycle for recombination (1-indexed).
        recomb_end (int, optional): Ending cycle for recombination (1-indexed, inclusive).
        recomb_col (str, optional): Name for recombination column. Default is 'prefix_recomb'.
        recomb_filter_col (str, optional): Quality column for filtering recombination calls.
        recomb_q_thresh (float, optional): Minimum quality for recombination. Default is 0.1.

        sort_calls (str, optional): Sorting criterion - 'count' or 'peak'. Default is 'peak'.
        error_correct (bool, optional): Whether to perform error correction. Default is False.
        max_distance (int, optional): Maximum edit distance for correction. Default is 2.
        distance_metric (str, optional): 'hamming' or 'levenshtein'. Default is 'hamming'.

        barcode_info_cols (list, optional): Columns from library to merge.
            Defaults to [GENE_SYMBOL, GENE_ID] if available.
        df_UMI (DataFrame, optional): DataFrame with UMI reads for UMI support.

        kwargs: Additional arguments for error correction.

    Returns:
        DataFrame: Standardized output with columns:
            - well, tile, cell
            - Q_min, Q_0, Q_1, ... (quality scores)
            - cell_barcode_0, cell_barcode_1 (ALWAYS named cell_barcode, never sgRNA)
            - barcode_peak_0, barcode_peak_1 (always present)
            - barcode_count_0, barcode_count_1, barcode_count (if sort_calls="count")
            - no_recomb_0, no_recomb_1 (if recombination enabled, else NaN)
            - pre_correction_cell_barcode_0, pre_correction_cell_barcode_1
              (if error_correct=True)
            - gene_symbol_0, gene_symbol_1 (if library provided)
            - gene_id_0, gene_id_1 (if available in library)
            - UMI columns (if df_UMI provided)

    Examples:
        # Single-barcode with count sorting (original behavior)
        >>> call_cells(reads, library, barcode_col="sgRNA", sort_calls="count")

        # Single-barcode with peak sorting
        >>> call_cells(reads, library, barcode_col="sgRNA", sort_calls="peak")

        # Multi-barcode with recombination detection
        >>> call_cells(reads, library,
        ...           map_start=1, map_end=15,
        ...           recomb_start=16, recomb_end=30,
        ...           recomb_filter_col="Q_recomb",
        ...           sort_calls="peak")

        # Cycle-based for single-barcode (use full barcode range)
        >>> call_cells(reads, library,
        ...           map_start=1, map_end=15,
        ...           sort_calls="count")
    """
    # Handle empty input: Return standardized empty DataFrame when no reads are available
    # This occurs when: (1) reads_data is None, or (2) reads_data is an empty DataFrame
    # Returning an empty DataFrame with all expected columns prevents crashes in downstream
    # processing and allows pipelines to continue even when tiles/wells have no reads
    if reads_data is None or reads_data.empty:
        return _get_empty_output()

    cols = [WELL, TILE, CELL]

    # STEP 1: Determine barcode extraction mode and prepare reads
    # Three mutually exclusive modes determine how barcodes are extracted:
    # 1. Cycle-based: Extract specific cycle ranges (for multi-barcode protocols)
    # 2. Pre-computed prefix: Use pre-existing prefix column from library
    # 3. Auto-truncation: Automatically truncate full barcodes to match read length

    if map_start is not None and map_end is not None:
        # CYCLE-BASED MODE: Extract barcodes from specific cycle ranges
        # Used for multi-barcode protocols or when explicit cycle specification is needed
        print(f"Using cycle-based extraction: map cycles {map_start}-{map_end}")
        df_reads = prep_multi_reads(
            reads_data,
            map_start=map_start,
            map_end=map_end,
            recomb_start=recomb_start or map_start,  # Default to map range
            recomb_end=recomb_end or map_end,
            map_col=map_col,
            recomb_col=recomb_col,
        )
        barcode_column = map_col
        enable_recomb = recomb_start is not None and recomb_col in df_reads.columns
        library_key = map_col

    elif prefix_col is not None:
        # PRE-COMPUTED PREFIX MODE: Use pre-existing prefixes from library
        # Useful when prefixes have already been computed (e.g., with cycle skipping)
        if (
            df_barcode_library is not None
            and prefix_col not in df_barcode_library.columns
        ):
            raise ValueError(f"Column '{prefix_col}' not found in barcode library")
        print(f"Using pre-computed prefixes from '{prefix_col}' column")
        df_reads = reads_data
        barcode_column = BARCODE
        enable_recomb = False
        library_key = PREFIX
        if df_barcode_library is not None:
            df_barcode_library[PREFIX] = df_barcode_library[prefix_col]

    else:
        # AUTO-TRUNCATION MODE: Automatically match barcode length to read length
        # Default mode - truncates full library barcodes to match experimental read length
        df_reads = reads_data
        barcode_column = BARCODE
        enable_recomb = False
        library_key = PREFIX
        if df_barcode_library is not None:
            # Determine experimental prefix length from first read
            prefix_length = len(reads_data.iloc[0].barcode)
            df_barcode_library[PREFIX] = df_barcode_library.apply(
                lambda x: x[barcode_col][:prefix_length], axis=1
            )
            print(
                f"Created prefixes by truncating '{barcode_col}' to length {prefix_length}"
            )

    # STEP 2: Apply quality filtering
    # Filter reads to only include those meeting minimum quality threshold
    df_reads = df_reads.query("Q_min >= @q_min")

    # STEP 3: Call cells using appropriate strategy
    # Two strategies available:
    # 1. Without reference library: Call barcodes directly from reads
    # 2. With reference library: Map reads to known barcodes with optional error correction

    if df_barcode_library is None:
        # NO REFERENCE LIBRARY: Call barcodes directly without mapping to known library
        df_cells = _call_cells_no_ref(df_reads, barcode_column, sort_calls=sort_calls)

    else:
        # WITH REFERENCE LIBRARY: Map reads to known barcodes with optional annotations

        # Determine which columns to include from the barcode library
        # Default: include gene_symbol and gene_id if available
        if barcode_info_cols is None:
            # Default to gene_symbol and gene_id if available
            barcode_info_cols = []
            if GENE_SYMBOL in df_barcode_library.columns:
                barcode_info_cols.append(GENE_SYMBOL)
            if GENE_ID in df_barcode_library.columns:
                barcode_info_cols.append(GENE_ID)
            # Fallback to sgRNA if no gene columns
            if not barcode_info_cols and SGRNA in df_barcode_library.columns:
                barcode_info_cols.append(SGRNA)

        df_cells = _call_cells_mapping(
            df_reads,
            df_barcode_library,
            barcode_column=barcode_column,
            library_key=library_key,
            barcode_info_cols=barcode_info_cols,
            enable_recomb=enable_recomb,
            recomb_col=recomb_col if enable_recomb else None,
            recomb_filter_col=recomb_filter_col,
            recomb_q_thresh=recomb_q_thresh,
            error_correct=error_correct,
            sort_calls=sort_calls,
            max_distance=max_distance,
            distance_metric=distance_metric,
            **kwargs,
        )

    # STEP 4: Add UMI (Unique Molecular Identifier) information if available
    # UMIs can be added to track individual RNA molecules
    if df_UMI is not None:
        df_cells = call_cells_add_UMIs(df_cells, df_UMI, cols=cols)

    return df_cells

def call_cells_v2(
        df_reads,
        library_lookup,
        alpha=1.5,
        max_pair_dist=2,
        max_assign_dist=1):
    
    import numpy as np
    import pandas as pd
    from functools import lru_cache

    # 1. Distance + scoring utilities
    def hamming(a, b):
        """Hamming distance for equal-length strings"""
        return sum(x != y for x, y in zip(a, b))

    @lru_cache(None)
    def exp_score(d, alpha=1.5):
        """Soft penalty for sequencing errors"""
        return np.exp(-alpha * d)


    # 2. Pair-aware scoring
    def score_pair(cell_reads, pair, alpha=1.5, max_dist=2):
        """
        Score how well a library barcode pair explains
        all barcode observations in a cell.
        """
        b1, b2 = pair
        score = 0.0

        for bc in cell_reads["barcode"]:
            d1 = hamming(bc, b1)
            d2 = hamming(bc, b2)
            d = min(d1, d2)

            if d <= max_dist:
                score += exp_score(d, alpha)

        return score

    # 3. Assign best library pair per cell
    def assign_cell(cell_reads, lib_pairs, alpha=1.5, max_dist=2):
        scores = np.array([
            score_pair(cell_reads, pair, alpha, max_dist)
            for pair in lib_pairs
        ])

        best_idx = np.argmax(scores)
        best_pair = lib_pairs[best_idx]
        best_score = scores[best_idx]

        # Confidence metric (score separation)
        sorted_scores = np.sort(scores)
        confidence = (
            sorted_scores[-1] / (sorted_scores[-2] + 1e-6)
            if len(sorted_scores) > 1 else np.inf
        )

        return {
            "iBar1": best_pair[0],
            "iBar2": best_pair[1],
            "score": best_score,
            "confidence": confidence
        }

    # 4. Extract corrected barcodes per cell
    def extract_corrected_barcodes(cell_reads, assigned_pair, max_dist=1):
        """
        Assign each observed barcode to one of the two
        library barcodes if within max_dist.
        """
        b1 = assigned_pair["iBar1"]
        b2 = assigned_pair["iBar2"]

        assigned = []

        for bc in cell_reads["barcode"]:
            d1 = hamming(bc, b1)
            d2 = hamming(bc, b2)

            if min(d1, d2) <= max_dist:
                assigned.append(b1 if d1 <= d2 else b2)

        return assigned

    # 5. Full CropSeqMulti optical barcode assignment
    lib_pairs = list(library_lookup.keys())  # library_lookup is a dict mapping (iBar1, iBar2) to gene symbol

    results = []
    for cell, cell_reads in df_reads.groupby("cell"):
        assignment = assign_cell(
            cell_reads,
            lib_pairs,
            alpha=alpha,
            max_dist=max_pair_dist
        )

        corrected_bcs = extract_corrected_barcodes(
            cell_reads,
            assignment,
            max_dist=max_assign_dist
        )

        bc_counts = (
            pd.Series(corrected_bcs)
            .value_counts()
            .to_dict()
        )

        results.append({
            "plate": cell_reads.iloc[0]["plate"],
            "well": cell_reads.iloc[0]["well"],
            "tile": cell_reads.iloc[0]["tile"],
            "cell": cell,
            "iBar1": assignment["iBar1"],
            "iBar2": assignment["iBar2"],
            "score": assignment["score"],
            "confidence": assignment["confidence"],
            "barcode_counts": bc_counts,
            "n_spots": len(cell_reads),
            "n_explained_spots": sum(bc_counts.values())
        })

    df_assignments = pd.DataFrame(results)

    # finally we can merge back to the original df_reads to get tile/well info and quality metrics
    def extract_pair_counts(barcode_counts, iBar1, iBar2):
        c0 = barcode_counts.get(iBar1, 0)
        c1 = barcode_counts.get(iBar2, 0)
        total = c0 + c1
        return c0, c1, total

    cell_rows = []

    for _, r in df_assignments.iterrows():
        c0, c1, total = extract_pair_counts(
            r.barcode_counts, r.iBar1, r.iBar2
        )

        g0 = library_lookup.get((r.iBar1, r.iBar2), None)

        cell_rows.append({
            "plate": r.plate,
            "well": r.well,
            "tile": r.tile,
            "cell": r.cell,

            "cell_barcode_0": r.iBar1,
            "cell_barcode_count_0": c0,

            "cell_barcode_1": r.iBar2,
            "cell_barcode_count_1": c1,

            "barcode_count": total,
            "barcode_peak_0": c0 / total if total else 0.0,
            "barcode_peak_1": c1 / total if total else 0.0,

            "gene_symbol_0": g0,
            "gene_symbol_1": g0,

            "mapped_0": c0 > 0,
            "mapped_1": c1 > 0,
        })


    return pd.DataFrame(cell_rows)


def call_cells_v3(
    df_reads_,
    library_lookup,
    alpha=1.5,
    max_pair_dist=2,
    max_assign_dist=1,
    group_cols=("tile", "cell"),
    q_min=0,
    max_q_mismatches = 1,
    max_n_reads_in_cell = 20,
    max_n_unexplained_spots_in_cell = 2,
    min_n_spots_in_cell = 2
):
    """
    Assign reads to the best barcode pair per cell using Hamming-distance-weighted scores.
    df_reads: DataFrame of individual reads; must contain a 'barcode' column and the grouping columns.
    library_lookup: iterable/list of (b1, b2) barcode pair tuples representing the library.
    alpha: exponential penalty factor for pair distance (higher -> stronger penalty).
    max_pair_dist: max Hamming distance considered when scoring pair contributions.
    max_assign_dist: max Hamming distance allowed when assigning a read to one of the pair barcodes.
    group_cols: tuple of column names used to group reads into cells (prevents cross-tile mixing).
    q_min: minimum q value for a read to be valid 
    max_q_mismatches: how many letters in the code are allowed to be lower than q_min
    max_n_reads_in_cell: cells with number of reads higher than this value will be discarded - consider them wrong in some way, maybe segmentation did not work and it merged several together
    """
    lib_pairs = list(library_lookup)
    n_pairs = len(lib_pairs) # library length
    group_cols = list(group_cols)

    # Precompute exponential penalties
    exp_pair = np.exp(-alpha * np.arange(max_pair_dist + 1))

    # Cache: barcode -> (pair_indices, contribution_values)
    pair_score_cache = {}

    # Filter reads by quality: keep only those with at most max_q_mismatches positions with Q < q_min
    mask = df_reads_.apply(lambda row: sum(1 for i in range(15) if f'Q_{i}' in row and row[f'Q_{i}'] < q_min) <= max_q_mismatches, axis=1)
    df_reads = df_reads_[mask]
    
    # Filter out cells with excessive amount of reads (likely wrongful detection)
    reads_per_cell = df_reads.groupby(group_cols).size()
    cells_to_keep = reads_per_cell[reads_per_cell <= max_n_reads_in_cell].index
    df_reads = df_reads.set_index(group_cols).loc[cells_to_keep].reset_index()

    def hamming(a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    def get_pair_score_sparse(bc):
        """
        Returns sparse pair contribution vector for barcode `bc`.
        """
        cached = pair_score_cache.get(bc)
        if cached is not None:
            return cached

        idx = []
        vals = []

        for k, (b1, b2) in enumerate(lib_pairs):
            d = min(hamming(bc, b1), hamming(bc, b2))
            if d <= max_pair_dist:
                idx.append(k)
                vals.append(exp_pair[d])

        idx = np.array(idx, dtype=np.int32)
        vals = np.array(vals, dtype=np.float32)

        pair_score_cache[bc] = (idx, vals)
        return idx, vals

    # Iterate for each cell 
    #    i.e.roup by plate + well + tile + cell to avoid cross-tile collisions
    results = []

    for keys, grp in df_reads.groupby(group_cols, sort=False):

        if isinstance(keys, tuple):
            key_dict = dict(zip(group_cols, keys))
        else:
            key_dict = {group_cols[0]: keys}

        # Extract all found barcodes in a cell and count how many times each of them
        obs = grp["barcode"].to_numpy()
        uniq, counts = np.unique(obs, return_counts=True)

        scores = np.zeros(n_pairs, dtype=float)

        # Accumulate contributions per unique barcode
        for bc, ct in zip(uniq, counts):
            idx, vals = get_pair_score_sparse(bc)
            if len(idx) > 0:
                scores[idx] += ct * vals

        # Best pair
        best_idx = int(np.argmax(scores))
        best_pair = lib_pairs[best_idx]
        best_score = float(scores[best_idx])

        if n_pairs > 1:
            top2 = np.partition(scores, -2)[-2:]
            confidence = float(top2.max() / (top2.min() + 1e-6))
        else:
            confidence = np.inf

        # Corrected barcode assignment (reuse distances minimally)
        b1, b2 = best_pair
        corrected_counts = {}
        explained = 0

        for bc, ct in zip(uniq, counts):
            d1 = hamming(bc, b1)
            d2 = hamming(bc, b2)

            if min(d1, d2) <= max_assign_dist:
                assigned = b1 if d1 <= d2 else b2
                corrected_counts[assigned] = (
                    corrected_counts.get(assigned, 0) + int(ct)
                )
                explained += int(ct)
        
        n_spots = int(len(obs))
        # Discard cell if not enough explained spots 
        if explained < min_n_spots_in_cell:
            continue
        # Discard cell if too many unexplained spots, given the assigned barcode pair
        if n_spots - explained > max_n_unexplained_spots_in_cell: 
            continue

        results.append({
            "plate": grp.iloc[0]["plate"],
            "well": grp.iloc[0]["well"],
            "tile": grp.iloc[0]["tile"],
            "cell": grp.iloc[0]["cell"],
            "iBar1": b1,
            "iBar2": b2,
            "score": best_score,
            "confidence": confidence,
            "barcode_counts": corrected_counts,
            "n_spots": n_spots,
            "n_explained_spots": int(explained),
        })

    df_assignments = pd.DataFrame(results)
    
    # finally we can merge back to the original df_reads to get tile/well info and quality metrics
    def extract_pair_counts(barcode_counts, iBar1, iBar2):
        c0 = barcode_counts.get(iBar1, 0)
        c1 = barcode_counts.get(iBar2, 0)
        total = c0 + c1
        return c0, c1, total

    cell_rows = []

    for _, r in df_assignments.iterrows():
        c0, c1, total = extract_pair_counts(
            r.barcode_counts, r.iBar1, r.iBar2
        )

        g0 = library_lookup.get((r.iBar1, r.iBar2), None)

        cell_rows.append({
            "plate": r.plate,
            "well": r.well,
            "tile": int(r.tile),
            "cell": int(r.cell),

            "cell_barcode_0": r.iBar1,
            "cell_barcode_count_0": c0,

            "cell_barcode_1": r.iBar2,
            "cell_barcode_count_1": c1,
            "barcode_count": total,

            "gene_symbol_0": g0,
            "gene_symbol_1": g0,

            "mapped_0": c0 > 0,
            "mapped_1": c1 > 0,
        })


    return pd.DataFrame(cell_rows)


def _get_empty_output():
    """Return empty DataFrame with standardized column names for empty input handling."""
    columns = [
        "cell",
        "tile",
        "well",
        "Q_min",
        "peak",
        "cell_barcode_0",
        "barcode_peak_0",
        "cell_barcode_1",
        "barcode_peak_1",
        "barcode_count_0",
        "barcode_count_1",
        "barcode_count",
        "no_recomb_0",
        "no_recomb_1",
        "gene_symbol_0",
        "gene_symbol_1",
        "gene_id_0",
        "gene_id_1",
    ]
    return pd.DataFrame(columns=columns)


def _call_cells_no_ref(df_reads, barcode_column, sort_calls="peak"):
    """Call cells without reference library.

    Identifies the most abundant barcodes in each cell without mapping to a known
    library. Supports both count-based and peak-based sorting strategies.

    Args:
        df_reads (DataFrame): Filtered read data
        barcode_column (str): Name of barcode column to use
        sort_calls (str): "count" or "peak"

    Returns:
        DataFrame: Cell calls with standardized column names
    """
    cols = [WELL, TILE, CELL]

    if sort_calls == "count":
        # COUNT-BASED SORTING: Prioritize barcodes by number of reads
        # Best for mRNA barcodes where multiple spots per cell are expected

        # Create grouped ranking: count barcodes per cell and sort by frequency
        # s.nth(0) = most frequent barcode per cell, s.nth(1) = second most frequent
        s = (
            df_reads.drop_duplicates([WELL, TILE, READ])  # Remove duplicate reads
            .groupby(cols)[barcode_column]  # Group by (well, tile, cell)
            .value_counts()  # Count each barcode's frequency per cell
            .rename("count")
            .sort_values(ascending=False)  # Sort so top barcodes come first
            .reset_index()
            .groupby(cols)  # Re-group by cell for .nth() access
        )

        # Build output dataframe by joining top 2 barcodes and their counts to df_reads
        # Multiple joins are used because each adds different columns (barcode vs count)
        # set_index(cols) enables joining on (well, tile, cell) identifiers
        df_cells = (
            df_reads.join(  # Join #1: Add top-ranked barcode as cell_barcode_0
                s.nth(0)[["well", "tile", "cell", barcode_column]]
                .rename(columns={barcode_column: BARCODE_0})
                .set_index(cols),  # Index by (well, tile, cell) for join
                on=cols,
            )
            .join(  # Join #2: Add count for top barcode as barcode_count_0
                s.nth(0)[["well", "tile", "cell", "count"]]
                .rename(columns={"count": BARCODE_COUNT_0})
                .set_index(cols),
                on=cols,
            )
            .join(  # Join #3: Add second-ranked barcode as cell_barcode_1
                s.nth(1)[["well", "tile", "cell", barcode_column]]
                .rename(columns={barcode_column: BARCODE_1})
                .set_index(cols),
                on=cols,
            )
            .join(  # Join #4: Add count for second barcode as barcode_count_1
                s.nth(1)[["well", "tile", "cell", "count"]]
                .rename(columns={"count": BARCODE_COUNT_1})
                .set_index(cols),
                on=cols,
            )
            .join(  # Join #5: Add total count summed across all barcodes
                s["count"].sum().rename(BARCODE_COUNT), on=cols
            )
            .assign(  # Fill NaN counts with 0 (cells with no/one barcode)
                **{
                    BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                    BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0),
                }
            )
            .drop_duplicates(cols)  # Keep one row per cell
            .drop(
                [READ, BARCODE, barcode_column], axis=1, errors="ignore"
            )  # Remove read-level columns
            .drop(
                [POSITION_I, POSITION_J], axis=1, errors="ignore"
            )  # Remove position columns
            .filter(regex="^(?!Q_)")  # Remove per-cycle quality columns
            .query("cell > 0")  # Filter to valid cells only
        )

        # Peak intensity not calculated in count mode - set to NaN
        df_cells["barcode_peak_0"] = np.nan
        df_cells["barcode_peak_1"] = np.nan

    else:
        # PEAK-BASED SORTING: Prioritize barcodes by peak intensity
        # Best for DNA barcodes with single bright spots per cell

        # Create grouped ranking: sort reads by peak intensity within each cell
        # s.nth(0) = brightest spot per cell, s.nth(1) = second brightest
        s = df_reads.sort_values("peak", ascending=False).groupby(cols)

        # Build output dataframe by joining top 2 barcodes and their peak intensities
        # Simpler than count mode since both barcode and peak come from same row
        df_cells = (
            df_reads.join(  # Join #1: Add brightest barcode and its peak intensity
                s.nth(0)[cols + [barcode_column, "peak"]]
                .rename(columns={barcode_column: BARCODE_0, "peak": "barcode_peak_0"})
                .set_index(cols),  # Index by (well, tile, cell) for join
                on=cols,
            )
            .join(  # Join #2: Add second brightest barcode and its peak intensity
                s.nth(1)[cols + [barcode_column, "peak"]]
                .rename(columns={barcode_column: BARCODE_1, "peak": "barcode_peak_1"})
                .set_index(cols),
                on=cols,
            )
            .drop_duplicates(cols)  # Keep one row per cell
            .drop(
                [READ, BARCODE, barcode_column, "peak"], axis=1, errors="ignore"
            )  # Remove read-level columns
            .drop(
                [POSITION_I, POSITION_J], axis=1, errors="ignore"
            )  # Remove position columns
            .filter(regex="^(?!Q_)")  # Remove per-cycle quality columns
            .query("cell > 0")  # Filter to valid cells only
        )

        # Read counts not calculated in peak mode - set to NaN
        df_cells[BARCODE_COUNT_0] = np.nan
        df_cells[BARCODE_COUNT_1] = np.nan
        df_cells[BARCODE_COUNT] = np.nan

    # Add recombination columns as NaN (no library to detect recombination)
    df_cells["no_recomb_0"] = np.nan
    df_cells["no_recomb_1"] = np.nan

    return df_cells


def _call_cells_mapping(
    df_reads,
    df_barcode_library,
    barcode_column,
    library_key,
    barcode_info_cols,
    enable_recomb,
    recomb_col,
    recomb_filter_col,
    recomb_q_thresh,
    error_correct,
    sort_calls,
    max_distance,
    distance_metric,
    **kwargs,
):
    """Call cells with reference library mapping.

    Maps sequencing reads to a known barcode library, enabling gene annotation and
    quality control. Provides comprehensive support for error correction, recombination
    detection, and flexible sorting strategies.

    Features:
    - Error correction with pre-correction tracking for QC
    - Recombination detection for multi-barcode protocols
    - Both count and peak-based sorting strategies
    - Configurable barcode info columns from library

    Args:
        df_reads (DataFrame): Filtered read data
        df_barcode_library (DataFrame): Reference barcode library
        barcode_column (str): Name of barcode column in reads
        library_key (str): Name of key column in library (PREFIX or map_col)
        barcode_info_cols (list): Columns to merge from library
        enable_recomb (bool): Whether to perform recombination detection
        recomb_col (str): Name of recombination column
        recomb_filter_col (str): Quality column for filtering recombination
        recomb_q_thresh (float): Quality threshold for recombination
        error_correct (bool): Whether to perform error correction
        sort_calls (str): "count" or "peak"
        max_distance (int): Max edit distance for error correction
        distance_metric (str): "hamming" or "levenshtein"
        kwargs: Additional arguments for error correction

    Returns:
        DataFrame: Cell calls with gene info and standardized columns
    """
    cols = [WELL, TILE, CELL]
    pre_correct_col = None

    # OPTIONAL: Error correction
    # Correct sequencing errors by mapping reads to closest library barcode
    # Preserves original values for QC tracking
    if error_correct:
        print("performing error correction")
        pre_correct_col = f"pre_correction_{barcode_column}"
        # Store original values before correction
        df_reads[pre_correct_col] = df_reads[barcode_column]
        # Perform error correction
        df_reads[barcode_column] = error_correct_reads(
            df_reads[barcode_column],
            df_barcode_library[library_key],
            max_distance=max_distance,
            distance_metric=distance_metric,
            **kwargs,
        )

    # Map reads to reference library
    # Left join reads to library - unmapped reads will have NaN for library columns
    df_barcode_library["_temp_key"] = df_barcode_library[library_key]
    df_mapped = (
        pd.merge(
            df_reads,
            df_barcode_library[["_temp_key"]],
            how="left",
            left_on=barcode_column,
            right_on="_temp_key",
        )
        .assign(mapped=lambda x: pd.notnull(x["_temp_key"]))
        .drop("_temp_key", axis=1)
    )

    # OPTIONAL: Recombination detection (multi-barcode protocols only)
    # Detect and flag recombination events between MAP and RECOMB barcode regions
    if enable_recomb and recomb_col is not None:
        # Create mapping of expected recombination values
        recomb_map = df_barcode_library.set_index(library_key)[recomb_col].to_dict()
        # Flag sequences where actual matches expected
        df_mapped["no_recomb"] = (df_mapped[barcode_column].map(recomb_map)) == (
            df_mapped[recomb_col]
        )
        # Unmapped cells have undetermined recombination status
        df_mapped.loc[~df_mapped.mapped, "no_recomb"] = np.nan
        # Drop the recomb barcode column (we only need the boolean)
        df_mapped = df_mapped.drop(columns=[recomb_col], errors="ignore")

        # Apply quality threshold for recombination status if specified
        if recomb_filter_col is not None:
            df_mapped.loc[
                df_mapped[recomb_filter_col] < recomb_q_thresh, "no_recomb"
            ] = np.nan
    else:
        # No recombination detection - set to NaN
        df_mapped["no_recomb"] = np.nan

    # Sort and prioritize barcodes per cell
    # Two strategies: count (for mRNA) or peak (for DNA)

    if sort_calls == "count":
        # COUNT-BASED SORTING: Prioritize by number of reads per barcode
        # Prioritizes mapped barcodes over unmapped ones

        # Create grouped ranking: count barcodes per cell and sort by (mapped, count)
        # This ensures mapped barcodes are always ranked higher than unmapped ones
        # s.nth(0) = top-ranked barcode per cell, s.nth(1) = second-ranked
        s = (
            df_mapped.drop_duplicates([WELL, TILE, READ])  # Remove duplicate reads
            .groupby(cols + ["mapped"])[
                barcode_column
            ]  # Group by (well, tile, cell, mapped)
            .value_counts()  # Count each barcode's frequency per cell
            .rename("count")
            .reset_index()
            .sort_values(
                ["mapped", "count"], ascending=False
            )  # Mapped=True first, then by count
            .groupby(cols)  # Re-group by cell for .nth() access
        )

        # Build output by joining top 2 barcodes and their counts to df_reads
        # Multiple joins needed because each adds different columns
        # Note: Joins are to df_reads (not df_mapped) to preserve original read data
        if error_correct and pre_correct_col:
            # Error correction is enabled - include pre-correction column
            # Pre-correction columns are handled separately in peak mode below
            df_cells = (
                df_reads.join(  # Join #1: Add top-ranked barcode as cell_barcode_0
                    s.nth(0)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_0})
                    .set_index(cols),  # Index by (well, tile, cell) for join
                    on=cols,
                )
                .join(  # Join #2: Add count for top barcode as barcode_count_0
                    s.nth(0)[["well", "tile", "cell", "count"]]
                    .rename(columns={"count": BARCODE_COUNT_0})
                    .set_index(cols),
                    on=cols,
                )
                .join(  # Join #3: Add second-ranked barcode as cell_barcode_1
                    s.nth(1)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_1})
                    .set_index(cols),
                    on=cols,
                )
                .join(  # Join #4: Add count for second barcode as barcode_count_1
                    s.nth(1)[["well", "tile", "cell", "count"]]
                    .rename(columns={"count": BARCODE_COUNT_1})
                    .set_index(cols),
                    on=cols,
                )
                .join(  # Join #5: Add total count summed across all barcodes
                    s["count"].sum().rename(BARCODE_COUNT), on=cols
                )
            )
        else:
            # Standard mode without error correction
            df_cells = (
                df_reads.join(  # Join #1: Add top-ranked barcode as cell_barcode_0
                    s.nth(0)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_0})
                    .set_index(cols),  # Index by (well, tile, cell) for join
                    on=cols,
                )
                .join(  # Join #2: Add count for top barcode as barcode_count_0
                    s.nth(0)[["well", "tile", "cell", "count"]]
                    .rename(columns={"count": BARCODE_COUNT_0})
                    .set_index(cols),
                    on=cols,
                )
                .join(  # Join #3: Add second-ranked barcode as cell_barcode_1
                    s.nth(1)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_1})
                    .set_index(cols),
                    on=cols,
                )
                .join(  # Join #4: Add count for second barcode as barcode_count_1
                    s.nth(1)[["well", "tile", "cell", "count"]]
                    .rename(columns={"count": BARCODE_COUNT_1})
                    .set_index(cols),
                    on=cols,
                )
                .join(  # Join #5: Add total count summed across all barcodes
                    s["count"].sum().rename(BARCODE_COUNT), on=cols
                )
            )

        df_cells = df_cells.assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0),
            }
        )

        # Peak intensity not calculated in count mode - set to NaN
        df_cells["barcode_peak_0"] = np.nan
        df_cells["barcode_peak_1"] = np.nan

    else:
        # PEAK-BASED SORTING: Prioritize by peak intensity
        # Prioritizes mapped barcodes first, then by peak intensity within each group

        # Create grouped ranking: sort by (mapped, peak) within each cell
        # This ensures mapped barcodes are always ranked higher than unmapped ones
        # s.nth(0) = brightest (mapped) spot per cell, s.nth(1) = second brightest
        s = (
            df_mapped.drop_duplicates([WELL, TILE, READ])  # Remove duplicate reads
            .sort_values(
                ["mapped", "peak"], ascending=[False, False]
            )  # Mapped=True first, then by peak
            .groupby(cols)  # Group by (well, tile, cell) for .nth() access
        )

        # Build output by joining top 2 barcodes with their peaks and recombination status
        # Unlike count mode, all columns come from same row, so fewer joins needed
        # Note: Joins are to df_reads (not df_mapped) to preserve original read data
        if error_correct and pre_correct_col:
            # Error correction is enabled - include pre-correction barcode values
            df_cells = df_reads.join(  # Join #1: Add top-ranked barcode with peak, recomb status, and pre-correction barcode
                s.nth(0)[cols + [barcode_column, pre_correct_col, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_0,
                        pre_correct_col: f"pre_correction_{BARCODE_0}",
                        "no_recomb": "no_recomb_0",
                        "peak": "barcode_peak_0",
                    }
                )
                .set_index(cols),  # Index by (well, tile, cell) for join
                on=cols,
            ).join(  # Join #2: Add second-ranked barcode with peak and recomb status
                s.nth(1)[cols + [barcode_column, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_1,
                        "no_recomb": "no_recomb_1",
                        "peak": "barcode_peak_1",
                    }
                )
                .set_index(cols),
                on=cols,
            )
        else:
            # Standard mode without error correction
            df_cells = df_reads.join(  # Join #1: Add top-ranked barcode with peak and recomb status
                s.nth(0)[cols + [barcode_column, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_0,
                        "no_recomb": "no_recomb_0",
                        "peak": "barcode_peak_0",
                    }
                )
                .set_index(cols),  # Index by (well, tile, cell) for join
                on=cols,
            ).join(  # Join #2: Add second-ranked barcode with peak and recomb status
                s.nth(1)[cols + [barcode_column, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_1,
                        "no_recomb": "no_recomb_1",
                        "peak": "barcode_peak_1",
                    }
                )
                .set_index(cols),
                on=cols,
            )

        # Read counts not calculated in peak mode - set to NaN
        df_cells[BARCODE_COUNT_0] = np.nan
        df_cells[BARCODE_COUNT_1] = np.nan
        df_cells[BARCODE_COUNT] = np.nan

    # Clean up temporary columns and filter to cells only
    df_cells = (
        df_cells.drop_duplicates(cols)  # Keep one row per cell
        .drop(
            [READ, BARCODE, barcode_column], axis=1, errors="ignore"
        )  # Remove read-level columns
        .drop(
            [POSITION_I, POSITION_J], axis=1, errors="ignore"
        )  # Remove position columns
        .drop(
            ["no_recomb"], axis=1, errors="ignore"
        )  # Already split into no_recomb_0 and no_recomb_1
    )

    # Remove pre-correction temp column if it exists (already renamed to pre_correction_cell_barcode_0)
    if pre_correct_col and pre_correct_col in df_cells.columns:
        df_cells = df_cells.drop([pre_correct_col], axis=1, errors="ignore")

    # Filter to valid cell IDs (cell > 0)
    df_cells = df_cells.query("cell > 0")

    # Merge gene annotations from barcode library
    # Add gene_symbol, gene_id, and other annotations for both barcode_0 and barcode_1
    # Uses left merge so cells with unmapped barcodes still appear in output

    # First merge: Add annotations for primary barcode (cell_barcode_0)
    # Rename columns with _0 suffix to distinguish from secondary barcode
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[library_key] + barcode_info_cols],
            how="left",  # Keep all cells, even those without matching barcode
            left_on=BARCODE_0,  # Join on cell_barcode_0
            right_on=library_key,  # Match to library key (prefix or map_prefix)
        )
        .rename({col: col + "_0" for col in barcode_info_cols}, axis=1)  # Add _0 suffix
        .drop(library_key, axis=1, errors="ignore")  # Drop redundant library key column
    )

    # Second merge: Add annotations for secondary barcode (cell_barcode_1)
    # Rename columns with _1 suffix
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[library_key] + barcode_info_cols],
            how="left",  # Keep all cells, even those without matching barcode
            left_on=BARCODE_1,  # Join on cell_barcode_1
            right_on=library_key,  # Match to library key (prefix or map_prefix)
        )
        .rename({col: col + "_1" for col in barcode_info_cols}, axis=1)  # Add _1 suffix
        .drop(library_key, axis=1, errors="ignore")  # Drop redundant library key column
    )

    return df_cells


def prep_multi_reads(
    df_reads,
    map_start,
    map_end,
    recomb_start,
    recomb_end,
    map_col="prefix_map",
    recomb_col="prefix_recomb",
):
    """Prepare reads for multi-barcode calling by extracting cycle-specific barcodes.

    Multi-barcode protocols sequence two distinct barcode regions:
    - MAP region: Primary barcode for identifying perturbations
    - RECOMB region: Secondary barcode for detecting recombination events

    This function extracts specific cycle ranges from the full barcode sequence
    to create separate mapping and recombination barcodes. It also computes
    quality scores for the recombination region for filtering low-quality calls.

    Args:
        df_reads (DataFrame): DataFrame containing raw sequencing reads with columns:
            barcode, Q_0, Q_1, Q_2, ... (per-cycle quality scores)
        map_start (int): Starting cycle number for mapping barcode (1-indexed)
        map_end (int): Ending cycle number for mapping barcode (1-indexed, inclusive)
        recomb_start (int): Starting cycle for recombination barcode (1-indexed)
        recomb_end (int): Ending cycle for recombination barcode (1-indexed, inclusive)
        map_col (str, optional): Name for mapping barcode column. Default is 'prefix_map'
        recomb_col (str, optional): Name for recombination column. Default is 'prefix_recomb'

    Returns:
        DataFrame: Input DataFrame with added columns:
            - map_col: Extracted mapping barcode
            - recomb_col: Extracted recombination barcode
            - Q_recomb: Minimum quality score across recombination cycles
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df_reads.copy()

    # Handle empty DataFrame gracefully
    if df.empty:
        print(
            "Warning: DataFrame is empty, returning empty DataFrame with required columns"
        )
        df[map_col] = pd.Series(dtype="object")
        df[recomb_col] = pd.Series(dtype="object")
        df["Q_recomb"] = pd.Series(dtype="float64")
        return df

    # Check available quality columns
    available_q_cols = [
        col for col in df.columns if col.startswith("Q_") and col[2:].isdigit()
    ]
    max_cycle = (
        max([int(col[2:]) for col in available_q_cols]) + 1 if available_q_cols else 0
    )

    print(f"Available quality columns: {sorted(available_q_cols)}")
    print(f"Maximum cycle available: {max_cycle}")
    print(f"Requested mapping range: cycles {map_start}-{map_end}")
    print(f"Requested recombination range: cycles {recomb_start}-{recomb_end}")

    # Extract barcode subsequences from specified cycle ranges
    # Cycles are specified as 1-indexed, but string slicing is 0-indexed

    # Extract mapping barcode from specified cycles (convert to 0-indexing)
    df[map_col] = df["barcode"].str.slice(map_start - 1, map_end)

    # Extract recombination barcode from specified cycles (convert to 0-indexing)
    df[recomb_col] = df["barcode"].str.slice(recomb_start - 1, recomb_end)

    # Compute quality score for recombination region
    # Use minimum quality across recombination cycles for filtering
    recomb_cycles = list(range(recomb_start, recomb_end + 1))
    recomb_q_cols = [f"Q_{c - 1}" for c in recomb_cycles]

    # Check if all required quality columns exist
    missing_cols = [col for col in recomb_q_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing quality columns: {missing_cols}")
        available_cols = [col for col in recomb_q_cols if col in df.columns]
        if available_cols:
            df["Q_recomb"] = df[available_cols].min(axis=1)
        else:
            df["Q_recomb"] = pd.Series([np.nan] * len(df))
    else:
        df["Q_recomb"] = df[recomb_q_cols].min(axis=1)

    return df


def call_cells_add_UMIs(df_cells, df_UMI, cols=[WELL, TILE, CELL]):
    """Add UMI (Unique Molecular Identifier) information to called cells.

    Args:
        df_cells (DataFrame): DataFrame containing called cells
        df_UMI (DataFrame): DataFrame containing UMI reads with same structure as regular reads
        cols (list, optional): Columns for merging. Default is [WELL, TILE, CELL]

    Returns:
        DataFrame: df_cells with added UMI columns:
            - UMI_0, UMI_count_0: Top UMI and its count
            - UMI_1, UMI_count_1: Second UMI and its count
            - UMI_count: Total UMI count
    """
    s = (
        df_UMI.drop_duplicates([WELL, TILE, READ])
        .groupby(cols)[BARCODE]
        .value_counts()
        .rename("count")
        .sort_values(ascending=False)
        .reset_index()
        .groupby(cols)
    )

    df_cells_UMI = (
        df_UMI.join(
            s.nth(0)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": UMI_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": UMI_COUNT_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": UMI_1})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": UMI_COUNT_1})
            .set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(UMI_COUNT), on=cols)
        .assign(
            **{
                UMI_COUNT_0: lambda x: x[UMI_COUNT_0].fillna(0).astype(int),
                UMI_COUNT_1: lambda x: x[UMI_COUNT_1].fillna(0).astype(int),
            }
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE], axis=1, errors="ignore")
        .drop([POSITION_I, POSITION_J], axis=1, errors="ignore")
        .filter(regex="^(?!Q_)")
        .query("cell > 0")
    )

    cols_to_use = list(df_cells_UMI.columns.difference(df_cells.columns))

    return df_cells.merge(
        df_cells_UMI[cols_to_use + cols], left_on=cols, right_on=cols, how="inner"
    )


def error_correct_reads(reads, reference, max_distance=2, distance_metric="hamming"):
    """Error correct reads against a reference set of barcodes.

    Corrects sequencing errors by mapping each read to the closest reference barcode.
    Only corrects reads when:
    1. There is a unique closest match (no ties)
    2. The distance to that match is within max_distance threshold

    This conservative approach ensures we don't introduce incorrect corrections.

    Args:
        reads (pd.Series): Series with reads for error correction
        reference (pd.Series): Series with reference sequences
        max_distance (int, optional): Maximum distance for correction. Correction
            is performed only if (1) one reference sequence is closest (no ties)
            and (2) that unique reference is within this distance. Default is 2.
        distance_metric (str, optional): Distance metric to compare barcodes.
            Options are 'hamming' (default) and 'levenshtein'.

    Returns:
        pd.Series: Corrected reads (unchanged reads returned as-is)
    """
    # Calculate distance matrix: each read vs. each reference barcode
    dist_to_ref = barcode_distance_matrix(
        reads.to_list(),
        reference.to_list(),
        distance_metric=distance_metric,
    )

    # Find the minimum distance to any reference for each read
    min_dist_to_ref = dist_to_ref.min(axis=1)

    # Identify reads with a unique closest match (no ties)
    # True if exactly one reference barcode is at the minimum distance
    unique_dist = np.array(
        [
            np.sum(dist_to_ref[x] == min_dist_to_ref[x]) == 1
            for x in range(dist_to_ref.shape[0])
        ]
    )

    # Only correct reads that meet both criteria:
    # 1. Have a unique closest match
    # 2. Are within the maximum allowed distance
    corrected_subset = (unique_dist) & (min_dist_to_ref <= max_distance)

    # Get the corrected barcodes for eligible reads
    corrected_barcodes = reference.loc[
        dist_to_ref[corrected_subset].argmin(axis=1)
    ].values

    # Create a copy and update only the correctable reads
    corrected_reads = reads.copy()
    corrected_reads.loc[corrected_subset] = corrected_barcodes

    return corrected_reads


def barcode_distance_matrix(barcodes_1, barcodes_2=False, distance_metric="hamming"):
    """Calculate distances between two sets of barcodes.

    Creates a matrix of distances between all pairs of barcodes from two sets.
    If only one set is provided, computes self-distances.

    Args:
        barcodes_1 (list): First list of barcode sequences
        barcodes_2 (list or bool, optional): Second list of barcode sequences.
            If False, uses barcodes_1 for both sets. Default is False.
        distance_metric (str, optional): Type of distance to calculate.
            Options are 'hamming' or 'levenshtein'. Default is 'hamming'.

    Returns:
        numpy.ndarray: Matrix of distances between barcode pairs
    """
    import warnings

    # Define the distance function based on chosen metric
    if distance_metric == "hamming":
        distance = lambda i, j: Levenshtein.hamming(i, j)
    elif distance_metric == "levenshtein":
        distance = lambda i, j: Levenshtein.distance(i, j)
    else:
        warnings.warn(
            'distance_metric must be "hamming" or "levenshtein" - defaulting to "hamming"'
        )
        distance = lambda i, j: Levenshtein.hamming(i, j)

    # If second set not provided, use the first set
    if isinstance(barcodes_2, bool):
        barcodes_2 = barcodes_1

    # Create distance matrix for all barcode pairs
    bc_distance_matrix = np.zeros((len(barcodes_1), len(barcodes_2)))
    for a, i in enumerate(barcodes_1):
        for b, j in enumerate(barcodes_2):
            bc_distance_matrix[a, b] = distance(i, j)

    return bc_distance_matrix
