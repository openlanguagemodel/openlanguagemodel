pipeline(
    [
        layernorm(),
        linear_projection(),
        rope(),
        pipeline(
            [

            ]
        ), pipeline()
    ]
)