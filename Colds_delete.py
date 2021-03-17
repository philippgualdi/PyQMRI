import h5py

filename = "../23052018_VFA_89.h5"

h5 = h5py.File(filename, 'a')

if 'Coils' in h5:
    Coils = h5['Coils']  # VSTOXX futures data

    print(Coils)

    del h5['Coils']
print(list(h5.keys()))
if 'flip_angle(s)' in h5:
    print("Flip angle exists")
    data = h5['flip_angle(s)']

    del h5['flip_angle(s)']
    h5['fa'] = data

h5.close()