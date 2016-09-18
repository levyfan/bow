package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.feature.Feature;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import org.apache.commons.math3.linear.*;
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

    private static final double ZERO = 10e-10;
    private static final double EPS = 10e-6;

    public RealMatrix kissMe(List<BowImage> bowImages, Feature.Type type) {
        int length = bowImages.get(0).sp4[0].features.get(type).length;

        Multimap<String, Integer> idMap = ArrayListMultimap.create();
        IntStream.range(0, bowImages.size()).forEach(i -> idMap.put(bowImages.get(i).id, i));

        List<Object[]> objects = IntStream.range(0, bowImages.size()).parallel().mapToObj(i -> {
            RealMatrix positive = MatrixUtils.createRealMatrix(length, length);
            RealMatrix negative = MatrixUtils.createRealMatrix(length, length);

            BowImage x = bowImages.get(i);
            System.out.println("kissme: " + x.id);

            // positive
            long countPositive = idMap.get(x.id).stream().mapToLong(j -> {
                BowImage y = bowImages.get(j);
                if (Objects.equals(x.id, y.id) && j > i && !Objects.equals(x.cam, y.cam)) {
                    Pair<RealMatrix, Long> pair = km(x, y, type);
                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(positive, pair.getFirst());
                    return pair.getSecond();
                } else {
                    return 0;
                }
            }).sum();

            // negative
            long countNegative = ThreadLocalRandom.current().ints(5, 0, bowImages.size()).mapToLong(j -> {
                BowImage y = bowImages.get(j);
                if (!Objects.equals(x.id, y.id) && !Objects.equals(x.cam, y.cam)) {
                    Pair<RealMatrix, Long> pair = km(x, y, type);
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

        RealMatrix M = inv(positive).subtract(inv(negative));
        return validateCovMatrix(M);
    }

    private static RealMatrix validateCovMatrix(RealMatrix sig) {
        try {
            new CholeskyDecomposition(sig, 1.0e-5, 1.0e-10);
            return sig;
        } catch (NonPositiveDefiniteMatrixException | NonSymmetricMatrixException e) {
            if (e instanceof NonSymmetricMatrixException) {
                NonSymmetricMatrixException ee = (NonSymmetricMatrixException) e;
                System.out.println("(" + ee.getRow() + "," + ee.getColumn() + ")="
                        + sig.getEntry(ee.getRow(), ee.getColumn()));
                System.out.println("(" + ee.getColumn() + "," + ee.getRow() + ")="
                        + sig.getEntry(ee.getColumn(), ee.getRow()));
            }

            EigenDecomposition eigen = new EigenDecomposition(sig);
            RealMatrix v = eigen.getV();
            RealMatrix d = eigen.getD().copy();

            for (int n = 0; n < d.getColumnDimension(); n ++) {
                if (d.getEntry(n, n) <= ZERO) {
                    d.setEntry(n, n, EPS);
                }
            }

            sig = v.multiply(d).multiply(v.transpose());
            try {
                new CholeskyDecomposition(sig,
                        CholeskyDecomposition.DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD,
                        CholeskyDecomposition.DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD);
            } catch (Exception ee) {
                ee.printStackTrace();
            }
            return sig;
        }
    }

    private static RealMatrix inv(RealMatrix matrix) {
        try {
            return new LUDecomposition(matrix).getSolver().getInverse();
        } catch (SingularMatrixException e) {
            e.printStackTrace();
            return new SingularValueDecomposition(matrix).getSolver().getInverse();
        }
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
