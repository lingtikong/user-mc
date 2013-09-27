# Install/unInstall package classes in LAMMPS

if (test $1 = 1) then

  cp -p min_mc.h ..
  cp -p min_mc.cpp ..

elif (test $1 = 0) then

  rm ../min_mc.h
  rm ../min_mc.cpp

fi
