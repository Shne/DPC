# miro

## Optimizations

### Eye Pass
**No optimization**, is just tracing and saving to hashgrid

### Photon Pass
**Optimizations!** Currently is tracing AND looking up in two hashgrids (eye\_mps, scattering\_mps)
NEW WAY: do tracing and just save to array. upload array and hashgrids to GPU and do hashgrid lookup in kernel per photonThatHitsFigure in array

(might be an optimization to generate rays, so that they are kinda sorted with regard to direction. So that photon rays close to eachother in the array, hit something close in the scene. It might (MIGHT) help the cache on the GPU.)

### Final Pass
**Optimizations!** Currently is for each pixel (and corrosponding HitInfo, which has had its flux set in the photon pass) going through ALL scattering MPs, calculating their contribution and then the shading.
NEW WAY: parallelize over pixels AND scattering MPs. That is, a kernel for each pixel which then calls a kernel for each scattering MP.





## Implementation

.cu file (and .h) for each pass needing a kernel.
each file has a "setup" method receiving pointers to data and returning result. they do error-handling, uploading, and actual calling of the kernels.



## Other optimizations
In the final pass, try and avoid sqrt. See if there is a fast approximate version somewhere, somehow. Perhaps in cuda itself


## new optimizations
Photon Pass. Try without hashgrid, that is, looking at ALL MPs in parallel.
Photon Pass. Try using WARP_SIZE to divide each hashgrid list across single warp, so they all have the same depth (ish) and thus have no diverging conditionals in one warp

Final Pass. Sum reduction (see slide 21 in 06++performance...) 
Final Pass. concurrently copy memory and do calculations, (slide 43)
Final Pass. See if we can use strided access somewhere... (slide 47)
