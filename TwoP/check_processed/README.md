# Checking registration and stability
## Step by step guide
1. Open the python script called "first_step_checking_registered_bin". In there you should change the name, date, processed data location etc
2. It will give you a list of the frames from the binary that should be compared. Note it gives the start frames and end frames for each interval.
3. Open ImageJ and go to File>Open>check_bin.ijm. Here you can copy and paste the start and end slice frames into the variables start_slices and end_slices respectively. **Note** before running the code, check how much memory you have alocated to ImageJ (especially important for very big binary files) Go to Edit>Options>Memory&Threads. If this is less than the size of the binary, you will get an error
4. Also change the usual (animal, date, etc). 
5. Now just press run and wait for the binay to load
6. Once the binary has loaded, check it for the quality of the registration.
7. Now check the AVG tiffs for any Z drift/photobleaching/bubble forming etc. Do this for each experiment (so compare the first file with the second, then the third with the fourth and so on).
8. Finally compare the first tif with the last tif to see how good the stability is between experiments.


