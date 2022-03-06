def get_name(opts):
    name = '{}_{}_{}'.format(opts.data_name, opts.model_name, opts.trigger)
    if opts.mlmmdr_lamb > 0:
        name = name + '_mlmmdr_{}_{}'.format(opts.mlmmdr_lamb, opts.mlmmdr_layer)
    return name
