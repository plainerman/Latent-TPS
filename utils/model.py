from model.internal_flow import InternalFlow


def construct_model(args, dataset):
    if args.flow_type == 'internal_coords':
        model = InternalFlow(args, dataset)
    else:
        raise NotImplementedError(f'flow {args.flow_type} not implemented')

    if args.double_precision:
        model = model.double()

    return model
