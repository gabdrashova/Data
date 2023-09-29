
// Define your variables
drive = "Z"; // Drive letter
animal = "Giuseppina"; // Animal name
date = "2023-01-24"; // Date
plane = "plane1"; // Plane name
processing_folder = "ProcessedData";
//processing_folder = "Suite2Pprocessedfiles";
dir =  drive + ":/"+processing_folder+"/" + animal + "/" + date + "/suite2p/" + plane + "/";
// Define your arrays for start and end times
start_slices = newArray(1, 12334, 12434, 28248, 28348, 36900, 37000, 75568, 75668, 82794, 82894, 114603); // Example: Starts at slices 1, 101, and 201
end_slices = newArray(100, 12433, 12533, 28347, 28447, 36999, 37099, 75667, 75767, 82893, 82993, 114702); // Example: Ends at slices 100, 200, and 300

// Open the bin
run("Raw...", "open="+dir+"data.bin image=[16-bit Signed] width=256 height=256 number=120000 little-endian");


// Loop over the arrays
for (i = 0; i < lengthOf(start_slices); i++) {
    start_slice = start_slices[i];
    end_slice = end_slices[i];

    // Convert start and end slices to strings and create the substack using variables
    //run("Make Substack...", "slices=" +start_slice+ "-" +end_slice"");
    run("Make Substack...", "slices="+start_slice+"-"+end_slice);
    run("Z Project...", "projection=[Average Intensity]");

    // Close the substack window
    selectWindow("Substack (" + start_slice + "-" + end_slice + ")"); // Adjust the window title if needed
    close();
    saveAs("Tiff", dir + "/AVG_tiffs/AVG_Substack (" + start_slice + "-" + end_slice + ").tif");
    selectWindow("data.bin");
}