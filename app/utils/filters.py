def format_number(value, format_type='number'):
    """Format numbers for display in templates"""
    if value is None:
        return '-'
        
    if format_type == 'currency':
        return f'${value:,.2f}'
    elif format_type == 'percentage':
        return f'{value:.1f}%'
    elif format_type == 'integer':
        return f'{int(value):,}'
    else:
        return f'{value:,.2f}' 