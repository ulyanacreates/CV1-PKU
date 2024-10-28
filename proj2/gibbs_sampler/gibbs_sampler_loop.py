

def run_optimized_loop(sweep, img_height, img_width, mask, distort, energy, beta, norm, name, size, ori):
    from gibbs_sampler_cython import optimized_loop
    optimized_loop(sweep, img_height, img_width, mask, distort, energy, beta, norm.encode('utf-8'), name.encode('utf-8'), size.encode('utf-8'), ori)