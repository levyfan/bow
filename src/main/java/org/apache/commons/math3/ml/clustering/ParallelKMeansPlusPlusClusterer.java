/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.commons.math3.ml.clustering;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.MathUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Clustering algorithm based on David Arthur and Sergei Vassilvitski k-means++ algorithm.
 *
 * @param <T> type of the points to cluster
 * @see <a href="http://en.wikipedia.org/wiki/K-means%2B%2B">K-means++ (wikipedia)</a>
 * @since 3.2
 */
public class ParallelKMeansPlusPlusClusterer<T extends Clusterable> extends KMeansPlusPlusClusterer<T> {

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     * <p>
     * The euclidean distance will be used as default distance measure.
     *
     * @param k the number of clusters to split the data into
     */
    public ParallelKMeansPlusPlusClusterer(final int k) {
        this(k, -1);
    }

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     * <p>
     * The euclidean distance will be used as default distance measure.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations) {
        this(k, maxIterations, new EuclideanDistance());
    }

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     * @param measure       the distance measure to use
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations, final DistanceMeasure measure) {
        this(k, maxIterations, measure, new JDKRandomGenerator());
    }

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     * @param measure       the distance measure to use
     * @param random        random generator to use for choosing initial centers
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations,
                                           final DistanceMeasure measure,
                                           final RandomGenerator random) {
        this(k, maxIterations, measure, random, EmptyClusterStrategy.LARGEST_VARIANCE);
    }

    /**
     * Build a clusterer.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     * @param measure       the distance measure to use
     * @param random        random generator to use for choosing initial centers
     * @param emptyStrategy strategy to use for handling empty clusters that
     *                      may appear during algorithm iterations
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations,
                                           final DistanceMeasure measure,
                                           final RandomGenerator random,
                                           final EmptyClusterStrategy emptyStrategy) {
        super(k, maxIterations, measure, random, emptyStrategy);
    }

    /**
     * Runs the K-means++ clustering algorithm.
     *
     * @param points the points to cluster
     * @return a list of clusters containing the points
     * @throws MathIllegalArgumentException if the data points are null or the number
     *                                      of clusters is larger than the number of data points
     * @throws ConvergenceException         if an empty cluster is encountered and the
     *                                      {@link #emptyStrategy} is set to {@code ERROR}
     */
    @Override
    public List<CentroidCluster<T>> cluster(final Collection<T> points)
            throws MathIllegalArgumentException, ConvergenceException {

        // sanity checks
        MathUtils.checkNotNull(points);

        // number of clusters has to be smaller or equal the number of data points
        if (points.size() < k) {
            throw new NumberIsTooSmallException(points.size(), k, false);
        }

        // create the initial clusters
        List<CentroidCluster<T>> clusters = chooseInitialCenters(points);

        // create an array containing the latest assignment of a point to a cluster
        // no need to initialize the array, as it will be filled with the first assignment
        int[] assignments = new int[points.size()];
        assignPointsToClusters(clusters, points, assignments);

        // iterate through updating the centers until we're done
        final int max = (maxIterations < 0) ? Integer.MAX_VALUE : maxIterations;
        for (int count = 0; count < max; count++) {
            AtomicBoolean emptyCluster = new AtomicBoolean(false);
            final List<CentroidCluster<T>> finalClusters = clusters;

            List<CentroidCluster<T>> newClusters = finalClusters.parallelStream().map(cluster -> {
                final Clusterable newCenter;
                if (cluster.getPoints().isEmpty()) {
                    newCenter = getClusterable(finalClusters);
                    emptyCluster.set(true);
                } else {
                    newCenter = centroidOf(cluster.getPoints(), cluster.getCenter().getPoint().length);
                }
                return new CentroidCluster<T>(newCenter);
            }).collect(Collectors.toList());

            int changes = assignPointsToClusters(newClusters, points, assignments);
            clusters = newClusters;

            // if there were no more changes in the point-to-cluster assignment
            // and there are no empty clusters left, return the current clusters
            if (changes == 0 && !emptyCluster.get()) {
                return clusters;
            }
        }
        return clusters;
    }

    /**
     * Use K-means++ to choose the initial centers.
     *
     * @param points the points to choose the initial centers from
     * @return the initial centers
     */
    @Override
    List<CentroidCluster<T>> chooseInitialCenters(final Collection<T> points) {

        // Convert to list for indexed access. Make it unmodifiable, since removal of items
        // would screw up the logic of this method.
        final List<T> pointList = Collections.unmodifiableList(new ArrayList<>(points));

        // The number of points in the list.
        final int numPoints = pointList.size();

        // Set the corresponding element in this array to indicate when
        // elements of pointList are no longer available.
        final boolean[] taken = new boolean[numPoints];

        // The resulting list of initial centers.
        final List<CentroidCluster<T>> resultSet = new ArrayList<>();

        // Choose one center uniformly at random from among the data points.
        final int firstPointIndex = random.nextInt(numPoints);

        final T firstPoint = pointList.get(firstPointIndex);

        resultSet.add(new CentroidCluster<>(firstPoint));

        // Must mark it as taken
        taken[firstPointIndex] = true;

        // To keep track of the minimum distance squared of elements of
        // pointList to elements of resultSet.
        final double[] minDistSquared = new double[numPoints];

        // Initialize the elements.  Since the only point in resultSet is firstPoint,
        // this is very easy.
        IntStream.range(0, numPoints).parallel().forEach(i -> {
            if (i != firstPointIndex) { // That point isn't considered
                double d = distance(firstPoint, pointList.get(i));
                minDistSquared[i] = d*d;
            }
        });

        while (resultSet.size() < k) {

            // Sum up the squared distances for the points in pointList not
            // already taken.
            double distSqSum = IntStream.range(0, numPoints).parallel()
                    .mapToDouble(i -> taken[i] ? 0.0 : minDistSquared[i])
                    .sum();

            // Add one new data point as a center. Each point x is chosen with
            // probability proportional to D(x)2
            final double r = random.nextDouble() * distSqSum;

            // The index of the next point to be added to the resultSet.
            int nextPointIndex = -1;

            // Sum through the squared min distances again, stopping when
            // sum >= r.
            double sum = 0.0;
            for (int i = 0; i < numPoints; i++) {
                if (!taken[i]) {
                    sum += minDistSquared[i];
                    if (sum >= r) {
                        nextPointIndex = i;
                        break;
                    }
                }
            }

            // If it's not set to >= 0, the point wasn't found in the previous
            // for loop, probably because distances are extremely small.  Just pick
            // the last available point.
            if (nextPointIndex == -1) {
                for (int i = numPoints - 1; i >= 0; i--) {
                    if (!taken[i]) {
                        nextPointIndex = i;
                        break;
                    }
                }
            }

            // We found one.
            if (nextPointIndex >= 0) {

                final T p = pointList.get(nextPointIndex);

                resultSet.add(new CentroidCluster<>(p));

                // Mark it as taken.
                taken[nextPointIndex] = true;

                if (resultSet.size() < k) {
                    // Now update elements of minDistSquared.  We only have to compute
                    // the distance to the new center to do this.
                    IntStream.range(0, numPoints).parallel().forEach(j -> {
                        // Only have to worry about the points still not taken.
                        if (!taken[j]) {
                            double d = distance(p, pointList.get(j));
                            double d2 = d * d;
                            if (d2 < minDistSquared[j]) {
                                minDistSquared[j] = d2;
                            }
                        }
                    });
                }

            } else {
                // None found --
                // Break from the while loop to prevent
                // an infinite loop.
                break;
            }
        }

        return resultSet;
    }
}
