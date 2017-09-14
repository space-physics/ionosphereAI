#!/usr/bin/env python
"""
used to patch "auroraldet.h5" that has false detections from clouds or sunrise
"""
from pathlib import Path
from shutil import copy2
import h5py
from matplotlib.pyplot import figure,draw,pause,show

def patchdet(infn, slic=None, vlim=None, quiet=False):

    infn = Path(infn).expanduser()

    if slic is None:
        plotdet(infn, vlim=vlim)
        return

    assert isinstance(slic,slice),'Must be a slice to patch file, or None to display contents'

    #if outfn is None or (outfn.is_file() and outfn.samefile(infn)):
    outfn = infn.parent/(infn.stem +'_patched.h5')

    print(f'copying {infn} to {outfn}, setting "detect" slice {slic} to 0')

    copy2(infn, outfn)

    with h5py.File(outfn,'r+') as f:
        f['/detect'][slic] = 0

    plotdet(infn,outfn, vlim, quiet)


def plotdet(infn,outfn=None, vlim=None, quiet=False):

    with h5py.File(infn,'r') as f:
        indet = f['/detect'][:]

        if 'preview' in f and not quiet:
            print('plotting movie of',infn)
            fg = figure()
            ax = fg.gca()

            if vlim is None:
                h = ax.imshow(f['/preview'][1])
            else:
                h = ax.imshow(f['/preview'][1],vmin=vlim[0],vmax=vlim[1])

            fg.colorbar(h,ax=ax)
            ht = ax.set_title('')

            # because they're scalars, they need to be .value instead of [:]
            Nfile = f['/nfile'].value
            decim = f['/previewDecim'].value
            step = f['/framestep'].value

            for i,I in enumerate(f['/preview']):
                h.set_data(I)
                ht.set_text(f'{step*decim*i} / {Nfile}')
                draw(); pause(0.1)
# %%
    if outfn is None:
        return

    with h5py.File(outfn,'r') as f:
        outdet = f['/detect'][:]

    fg = figure()
    ax = fg.gca()
    ax.plot(indet,'b',label='original')
    ax.plot(outdet,'r',label='patched')
    ax.set_title(f'{outfn}  Auroral Detections')
    ax.legend()

    outimg = outfn.with_suffix('.png')
    print('saving',outimg)

    fg.savefig(str(outimg), bbox_inches='tight',dpi=100)




if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('fn',help='HDF5 file to manuall patch over (remove false detections due to sunrise)')
    p.add_argument('-s','--startstop',help='length 1 or 2 (start,stop) slice',nargs='+',type=int)
    p.add_argument('-q','--quiet',help='dont show preview movie',action='store_true')
    p.add_argument('-vlim',help='preview brightness',nargs=2,type=int)
    p = p.parse_args()

    if p.startstop is not None:
        if len(p.startstop) == 1:
            slic = slice(p.startstop[0],None)
        elif len(p.startstop) == 2:
            slic = slice(p.startstop[0],p.startstop[1])
        else:
            raise ValueError('start or start stop')
    else:
        slic = None

    patchdet(p.fn, slic, p.vlim, p.quiet)

    show()