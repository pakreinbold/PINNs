def get_gradient(x, fun):    
    import tensorflow as tf
    # Take the derivative of function fun(x) wrt its only argument x    
    with tf.GradientTape() as tape:
        tape.watch(x)      
        y = fun(x)

    g = tape.gradient(y, x)
    return g