from __future__ import annotations


def levels_corr(raw_cell_levels_corr, unit_level, agg_level):
    # unit_level set in countries.json (ses_data_options), for agg_level user does not
    # necessarily know full name in levels_corr
    cell_levels_corr = raw_cell_levels_corr.copy().fillna("none")
    cell_levels_corr.columns = cell_levels_corr.columns.str.lower()
    agg_cols = cell_levels_corr.columns[
        cell_levels_corr.columns.str.startswith(agg_level.lower())
    ]
    try:
        agg_level = agg_cols[0]
    except IndexError as e:
        print(f"No match found for {agg_level} in cell_levels_corr")
        raise e

    cell_levels_corr = (
        cell_levels_corr.groupby([agg_level, unit_level.lower()])
        .first()
        .loc[:, agg_cols[1:]]
    )
    return cell_levels_corr
