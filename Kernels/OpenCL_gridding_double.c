

      #pragma OPENCL EXTENSION cl_khr_fp64: enable
      #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
      void AtomicAdd(__global double *val, double delta) {
        union {
          double f;
          ulong  i;
        } old;
        union {
          double f;
          ulong  i;
        } new;
        do {
          old.f = *val;
          new.f = old.f + delta;
        } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
      }

  __kernel void make_complex(__global double2 *out,__global double *re, __global double* im)
  {
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];

  }
  __kernel void deapo_adj(__global double2 *out, __global double2 *in, __constant double *deapo, const int dim, const double scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y] * deapo[x]* scale;

  }
  __kernel void deapo_fwd(__global double2 *out, __global double2 *in, __constant double *deapo, const int dim, const double scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y] * deapo[x] * scale;
  }

  __kernel void zero_tmp(__global double2 *tmp)
  {
    size_t x = get_global_id(0);
    tmp[x] = 0.0f;

  }

  __kernel void grid_lut(__global double *sg, __global double2 *s, __global double2 *kpos, const int gridsize, const double kwidth, __global double *dcf, __constant double* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;


    double* ptr, pti;
    double2 kdat = s[k+kDim*n+kDim*NDim*scan]*(double2)(dcf[k],dcf[k]);



    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;


    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(double)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(double)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;

             AtomicAdd(&(sg[2*(gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)]),(kern * kdat.s0));
             AtomicAdd(&(sg[2*(gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)+1]),(kern * kdat.s1));
  			    }
  			}
  		}
  }


  __kernel void invgrid_lut(__global double2 *s, __global double2 *sg, __global double2 *kpos, const int gridsize, const double kwidth, __global double *dcf, __constant double* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    double2 tmp_dat = 0.0f;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(double)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(double)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;
             tmp_dat += (double2)(kern,kern)*sg[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan];
  			    }
  			}
  		}
    s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(double2)(dcf[k],dcf[k]);
  }
__kernel void copy(__global double2 *out, __global double2 *in, const double scale)
  {
    size_t x = get_global_id(0);
    out[x] = in[x]*scale;

  }
  __kernel void masking(__global double2 *ksp, __global double *mask)
  {
    size_t x = get_global_id(0);
    ksp[x] = ksp[x]*mask[x];


  }