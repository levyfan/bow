package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class FeatureManager {

    private Feature[] features;

    public FeatureManager() throws IOException, URISyntaxException {
        this.features = new Feature[]{new Cn(), new Hog(), new Hsv(), new Siltp()};
    }

    public Map<Feature.Type, List<double[]>> feature(BowImage bowImage) {
        Map<Feature.Type, List<double[]>> featureMap = new EnumMap<>(Feature.Type.class);
        for (Feature feature : features) {
            featureMap.put(feature.name(), feature.extract(bowImage));
        }
        return featureMap;
    }

    public ListMultimap<Feature.Type, double[]> feature(File[] files, SuperPixelMethond spMethod) throws IOException {
        List<Map<Feature.Type, List<double[]>>> list = Lists.newArrayList(files)
                .parallelStream()
                .map(file -> {
                    try {
                        System.out.println(file.getName());
                        BufferedImage image = ImageIO.read(file);

                        BowImage bowImage = new BowImage(spMethod, null, image, null);
                        return feature(bowImage);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());

        ListMultimap<Feature.Type, double[]> m = ArrayListMultimap.create(
                Feature.Type.values().length, files.length + 1);
        for (Map<Feature.Type, List<double[]>> map : list) {
            for (Map.Entry<Feature.Type, List<double[]>> entry : map.entrySet()) {
                m.putAll(entry.getKey(), entry.getValue());
            }
        }
        return m;
    }
}
