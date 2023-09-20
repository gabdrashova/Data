/*
#@ String (visibility=MESSAGE, value="Please change all values", required=false) msg
#@ String (label="Drive") drive
#@ String (label="Animal") animal
#@ String (label="date YYYY-MM-DD") date
#@ String (label="plane") plane
#@ String (label="processing folder name") processing_folder
#@ String (label="start slices (comma-separated)") start_slices_input
#@ String (label="end slices (comma-separated)") end_slices_input
*/

// Define your variables
drive = "Z"; // Drive letter
animal = "Glaucus"; // Animal name
date = "2022-08-10"; // Date
plane = "plane1"; // Plane name
processing_folder = "ProcessedData";
//processing_folder = "Suite2Pprocessedfiles";
dir =  drive + ":/"+processing_folder+"/" + animal + "/" + date + "/suite2p/" + plane + "/";
// Define your arrays for start and end times
start_slices = newArray(1, 23799, 23899, 71323, 71423, 78549, 78649, 87200); // Example: Starts at slices 1, 101, and 201
end_slices = newArray(100, 23898, 23998, 71422, 71522, 78648, 78748, 87299); // Example: Ends at slices 100, 200, and 300

// Open the bin
run("Raw...", "open="+dir+"data.bin image=[16-bit Signed] width=256 height=256 number=874 little-endian");


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
    //saveAs("Tiff", dir + "/AVG_tiffs/AVG_Substack (" + start_slice + "-" + end_slice + ").tif");
    selectWindow("data.bin");
}