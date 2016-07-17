package com.github.levyfan.reid.viper;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

/**
 * @author fanliwen
 */
public class ViperTest {

    @Test
    public void test() throws URISyntaxException, IOException {
        File file = new File(this.getClass().getResource("/score.mat").toURI());
        MatFileReader reader = new MatFileReader(file);
        MLDouble mlDouble = (MLDouble) reader.getMLArray("score");

        RealMatrix score = MatrixUtils.createRealMatrix(mlDouble.getArray());

        double[] MR = new Viper().eval(score);
        Assert.assertTrue(Math.abs(MR[0] - 0.1049) < 0.001);
        Assert.assertTrue(Math.abs(MR[19] - 0.3954) < 0.001);
    }
}
