package cse512

object HotzoneUtils {

  def ST_Contains(queryRectangle: String, pointString: String ): Boolean = {
    // YOU NEED TO CHANGE THIS PART
    //splitting point x, y coordinates into p_x, p_y
    val p = pointString.split(",")
    val p_x = p(0).trim().toDouble
    val p_y = p(1).trim().toDouble

    //splitting rectangle x1, y1, x2, y2 coordinates into r_x_1, r_x_2, r_x_3, r_x_4
    val r = queryRectangle.split(",")
    val r_x_1 = r(0).trim().toDouble
    val r_y_1 = r(1).trim().toDouble
    val r_x_2 = r(2).trim().toDouble
    val r_y_2 = r(3).trim().toDouble

    //Find min and max values between x1 & x2
    var min_x: Double = 0
    var max_x: Double = 0
    if(r_x_1 < r_x_2) {
      min_x = r_x_1
      max_x = r_x_2
    }
    else {
      min_x = r_x_2
      max_x = r_x_1
    }
    //Find min and max values between y1 & y2
    var min_y: Double = 0
    var max_y: Double = 0
    if(r_y_1 < r_y_2) {
      min_y = r_y_1
      max_y = r_y_2
    } else {
      min_y = r_y_2
      max_y = r_y_1
    }
    // Checking if the given point's x-coordinate p_x falls between min_x & max_x and
    // y-coordinate p_y falls between min_y && max_y
    if(p_x >= min_x && p_x <= max_x && p_y >= min_y && p_y <= max_y) {
      return true
    } else {
      return false
    }
  }
  // YOU NEED TO CHANGE THIS PART

}