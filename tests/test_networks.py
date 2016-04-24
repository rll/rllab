def test_gru_network():
    from rllab.core.network import GRUNetwork
    import lasagne.layers as L
    from rllab.misc import ext
    import numpy as np
    network = GRUNetwork(
        input_shape=(2, 3),
        output_dim=5,
        hidden_dim=4,
    )
    f_output = ext.compile_function(
        inputs=[network.input_layer.input_var],
        outputs=L.get_output(network.output_layer)
    )
    assert f_output(np.zeros((6, 8, 2, 3))).shape == (6, 8, 5)
