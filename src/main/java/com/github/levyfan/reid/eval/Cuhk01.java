package com.github.levyfan.reid.eval;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileType;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;

/**
 * @author fanliwen
 */
public class Cuhk01 {

    public RealMatrix select;

    public Cuhk01() throws URISyntaxException, IOException {
        try (InputStream stream = this.getClass().getResourceAsStream("/randselect10_cuhk01.mat")) {
            MatFileReader reader = new MatFileReader(stream, MatFileType.Regular);
            MLDouble mlDouble = (MLDouble) reader.getMLArray("selectsample");
            this.select = MatrixUtils.createRealMatrix(mlDouble.getArray());
        }
    }
}
