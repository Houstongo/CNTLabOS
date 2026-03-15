SORT_FIELD_MAP = {
    'id': 'id',
    'id_num': 'id',
    'sample_id': 'sample_id',
    'growth_temp': 'growth_temp',
    'growth_time': 'growth_time',
    'al2o3_thickness': 'al2o3_thickness',
    'al2o3_power': 'al2o3_power',
    'fe_thickness': 'fe_thickness',
    'fe_power': 'fe_power',
    'ar_flow': 'ar_flow',
    'h2_flow': 'h2_flow',
    'c2h4_flow': 'c2h4_flow',
    'anneal_temp': 'anneal_temp',
    'anneal_time': 'anneal_time',
    'magnification': 'magnification',
    'diameter': 'diameter',
    'density': 'density',
    'alignment': 'alignment',
    'curvature': 'curvature',
    'source': 'source'
}

def resolve_sort(sort_by: str, order: str):
    safe_sort_by = sort_by if sort_by in SORT_FIELD_MAP else 'id'
    safe_order = 'ASC' if str(order).lower() == 'asc' else 'DESC'
    return safe_sort_by, SORT_FIELD_MAP[safe_sort_by], safe_order
