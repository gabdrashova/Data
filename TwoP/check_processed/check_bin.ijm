// Define your variables
drive = "Z"; // Drive letter
animal = "Lotho"; // Animal name
date = "2023-04-12"; // Date
plane = "plane1"; // Plane name
processing_folder = "ProcessedData";
dir =  drive + ":/"+processing_folder+"/" + animal + "/" + date + "/suite2p/" + plane + "/";
// Open the image
run("Raw...", "open="+dir+"data.bin image=[16-bit Signed] width=512 height=512 number=50000 little-endian");

// Define your arrays for start and end times
start_slices = newArray(1, 6716, 6816 ,26617, 26717, 30334 ); // Example: Starts at slices 1, 101, and 201
end_slices = newArray(100, 6815, 6915, 26716, 26816,30433); // Example: Ends at slices 100, 200, and 300

// Loop over the arrays
for (i = 0; i < lengthOf(start_slices); i++) {
    start_slice = start_slices[i];
    end_slice = end_slices[i];

    // Create the substack using variables
    run("Make Substack...", "slices=" + start_slice + "-" + end_slice);
    run("Z Project...", "projection=[Average Intensity]");
	// Close the substack window
    selectWindow("Substack ("+start_slice+"-"+end_slice+")"); // Adjust the window title if needed
    close();
    saveAs("Tiff", dir+"/AVG_tiffs/AVG_Substack ("+start_slice+"-"+end_slice+").tif");
    selectWindow("data.bin");
	

}
