package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixel;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * @author fanliwen
 */
abstract class Opq implements Feature {

    private RealMatrix R;

    private Opq() throws IOException {
        MatFileReader reader = new MatFileReader(new File("codebook_opq_500_20.mat"));
        this.R = MatrixUtils.from((MLNumericArray) reader.getMLArray("R"));
    }

    @Override
    public abstract Type name();

    @Override
    public void extract(BowImage bowImage) {
        for (SuperPixel sp : bowImage.sp4) {
            double[] feature = Doubles.concat(
                    sp.features.get(Type.HSV),
                    sp.features.get(Type.CN),
                    sp.features.get(Type.HOG),
                    sp.features.get(Type.SILTP),
                    new double[]{0, 0}
            );

            double[] rotate = R.preMultiply(feature);

            int from = 0, to = 0;
            switch (name()) {
                case OPQ_1:
                    from = 0;
                    to = rotate.length / 4;
                    break;
                case OPQ_2:
                    from = rotate.length / 4;
                    to = rotate.length / 2;
                    break;
                case OPQ_3:
                    from = rotate.length / 2;
                    to = rotate.length * 3 / 4;
                    break;
                case OPQ_4:
                    from = rotate.length * 3 / 4;
                    to = rotate.length;
                    break;
            }
            sp.features.put(name(), Arrays.copyOfRange(rotate, from, to));
        }
    }

    @Override
    public String toString() {
        return "Opq{}";
    }

    static class Opq_1 extends Opq {

        Opq_1() throws IOException {
        }

        @Override
        public Type name() {
            return Type.OPQ_1;
        }
    }

    static class Opq_2 extends Opq {

        Opq_2() throws IOException {
        }

        @Override
        public Type name() {
            return Type.OPQ_2;
        }
    }

    static class Opq_3 extends Opq {

        Opq_3() throws IOException {
        }

        @Override
        public Type name() {
            return Type.OPQ_3;
        }
    }

    static class Opq_4 extends Opq {

        Opq_4() throws IOException {
        }

        @Override
        public Type name() {
            return Type.OPQ_4;
        }
    }
}
