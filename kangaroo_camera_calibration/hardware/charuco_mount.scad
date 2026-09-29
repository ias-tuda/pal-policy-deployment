// Mount plate for the ChArUco calibration board (camera_extrinsic_calib.py board defaults:
// 4x5 squares of 30 mm = 120 x 150 mm). The printed board is glued into a shallow pocket;
// a tab on one short side carries two mounting holes, centered and 29 mm apart, so the
// screw heads do not cover the board. All dimensions in mm.

board_w     = 120;    // board size along the short side
board_h     = 150;    // board size along the long side
pocket_gap  = 0.5;    // clearance around the paper in the pocket
pocket_d    = 0.6;    // pocket depth (locates the paper; glue it flat)
rim         = 4;      // plate border around the pocket
plate_t     = 5;      // plate thickness

tab_w       = 45;     // tab width along the short side
tab_l       = 16;     // tab length beyond the plate edge
hole_pitch  = 29;     // center-to-center distance of the two holes
hole_d      = 3.4;    // M3 clearance; M4 -> 4.5
hole_off    = 8;      // hole centers from the outer edge of the tab
corner_r    = 3;

$fn = 64;

plate_w = board_w + 2 * pocket_gap + 2 * rim;
plate_h = board_h + 2 * pocket_gap + 2 * rim;

module rounded_rect(w, h, r) {
    offset(r) offset(-r) square([w, h], center = true);
}

difference() {
    linear_extrude(plate_t) union() {
        rounded_rect(plate_w, plate_h, corner_r);
        // tab on the -y short side, overlapping the plate so the union is one body
        translate([0, -plate_h / 2 - tab_l / 2 + corner_r / 2])
            rounded_rect(tab_w, tab_l + corner_r, corner_r);
    }
    // paper pocket on the top face
    translate([0, 0, plate_t - pocket_d])
        linear_extrude(pocket_d + 1)
            square([board_w + 2 * pocket_gap, board_h + 2 * pocket_gap], center = true);
    // mounting holes
    for (sx = [-1, 1])
        translate([sx * hole_pitch / 2, -plate_h / 2 - tab_l + hole_off, -1])
            cylinder(d = hole_d, h = plate_t + 2);
}
