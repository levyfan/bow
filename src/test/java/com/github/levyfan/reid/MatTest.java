package com.github.levyfan.reid;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class MatTest {

    @Test
    public void test() throws IOException {
        List<double[]> list = new ArrayList<>();
        list.add(new double[]{1,2});
        list.add(new double[]{3,4});
        list.add(new double[]{5,6});
        list.add(new double[]{7,8});

        double[] array = Doubles.concat(Iterables.toArray(list, double[].class));

        List<MLArray> mlArrays = Lists.newArrayList(new MLDouble("array", array, 2));
        new MatFileWriter().write(
                "test.mat",
                mlArrays);
    }
}
