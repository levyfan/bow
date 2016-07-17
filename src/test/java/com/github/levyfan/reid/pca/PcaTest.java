package com.github.levyfan.reid.pca;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class PcaTest {

    @Test
    public void test() {
        List<double[]> x = new ArrayList<>();
        x.add(new double[]{1,3,5,7});
        x.add(new double[]{2,4,6,8});

        Pca pca = new Pca(x);
        BlasPca blasPca = new BlasPca(x);

        Assert.assertTrue(Math.abs(pca.ux.getEntry(0, 0) - 1) <= 0.001);
        Assert.assertTrue(Math.abs(blasPca.ux.get(0, 0) - 1) <= 0.001);
        Assert.assertTrue(Math.abs(pca.ux.getEntry(0, 1) + 1) <= 0.001);
        Assert.assertTrue(Math.abs(blasPca.ux.get(0, 1) + 1) <= 0.001);

        Assert.assertTrue(Math.abs(pca.eigvalue[0] - 2) <= 0.001);
        Assert.assertTrue(Math.abs(blasPca.eigvalue[0] - 2) <= 0.001);
    }
}
