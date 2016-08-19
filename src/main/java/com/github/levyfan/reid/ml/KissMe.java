package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.feature.Feature;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class KissMe {

    public RealMatrix kissMe(List<BowImage> bowImages, Feature.Type type) {
        int length = bowImages.get(0).sp4[0].features.get(type).length;

        List<Object[]> objects = IntStream.range(0, bowImages.size()).parallel().mapToObj(i -> {
            RealMatrix positive = MatrixUtils.createRealMatrix(length, length);
            RealMatrix negative = MatrixUtils.createRealMatrix(length, length);

            BowImage x = bowImages.get(i);
            System.out.println("kissme: " + x.id);

            // positive
            long countPositive = 0;
            for (int j = i + 1; j < bowImages.size(); j++) {
                if (Objects.equals(x.id, bowImages.get(j).id)) {
                    Pair<RealMatrix, Long> pair = km(x, bowImages.get(j), type);
                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(positive, pair.getFirst());
                    countPositive += pair.getSecond();
                } else {
//                    break;
                }
            }

            // negative
            long countNegative = ThreadLocalRandom.current().ints(5, 0, bowImages.size()).mapToLong(j -> {
                if (!Objects.equals(x.id, bowImages.get(j).id)) {
                    Pair<RealMatrix, Long> pair = km(x, bowImages.get(j), type);
                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(negative, pair.getFirst());
                    return pair.getSecond();
                } else {
                    return 0;
                }
            }).sum();

            return new Object[] {positive, negative, countPositive, countNegative};
        }).collect(Collectors.toList());

        RealMatrix positive = MatrixUtils.createRealMatrix(length, length);
        RealMatrix negative = MatrixUtils.createRealMatrix(length, length);
        long countPositive = 0;
        long countNegative = 0;
        for (Object[] obj : objects) {
            com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(positive, (RealMatrix) obj[0]);
            countPositive += (long) obj[2];
            com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(negative, (RealMatrix) obj[1]);
            countNegative += (long) obj[3];
        }

        positive = positive.scalarMultiply(1.0/(double) countPositive);
        negative = negative.scalarMultiply(1.0/(double) countNegative);

        return new LUDecomposition(positive.subtract(negative)).getSolver().getInverse();
    }

    private static Pair<RealMatrix, Long> km(BowImage i, BowImage j, Feature.Type type) {
        int length = i.sp4[0].features.get(type).length;
        RealMatrix matrix = MatrixUtils.createRealMatrix(length, length);

        long count = 0;

        for (int nstrip = 0; nstrip < i.strip4.length; nstrip ++) {
            Strip iStrip = i.strip4[nstrip];
            Strip jStrip = j.strip4[nstrip];

            for (int iSuperPixel : iStrip.superPixels) {
                for (int jSuperPixel : jStrip.superPixels) {
                    ArrayRealVector iFeature = new ArrayRealVector(i.sp4[iSuperPixel].features.get(type), false);
                    ArrayRealVector jFeature = new ArrayRealVector(j.sp4[jSuperPixel].features.get(type), false);
                    ArrayRealVector delta = iFeature.subtract(jFeature);

                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(matrix, delta.outerProduct(delta));
                    count++;
                }
            }
        }
        return Pair.create(matrix, count);
    }
}
