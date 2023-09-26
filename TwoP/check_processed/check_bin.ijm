
// Define your variables
drive = "Z"; // Drive letter
animal = "Hedes"; // Animal name
date = "2022-08-05"; // Date
plane = "plane3"; // Plane name
processing_folder = "ProcessedData";
//processing_folder = "Suite2Pprocessedfiles";
dir =  drive + ":/"+processing_folder+"/" + animal + "/" + date + "/suite2p/" + plane + "/";
// Define your arrays for start and end times
start_slices = newArray(1, 47360, 47460, 94722, 94822, 101948, 102048, 110599, 110699); // Example: Starts at slices 1, 101, and 201
end_slices = newArray(100, 47459, 47559, 94821, 94921, 102047, 102147, 110698, 110798); // Example: Ends at slices 100, 200, and 300

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