package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.bow.StripMethod;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.sp.SuperPixel;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.github.levyfan.reid.util.ImageUtils;

import java.awt.image.BufferedImage;
import java.util.EnumMap;
import java.util.Map;

/**
 * @author fanliwen
 */
public class BowImage {

    public BufferedImage image;

    public BufferedImage image4;

    public BufferedImage mask;

    public BufferedImage mask4;

    public SuperPixel[] sp4;

    public Strip[] strip4;

    public String id;

    public Map<Feature.Type, double[]> hist = new EnumMap<>(Feature.Type.class);

//    public Map<Feature.Type, double[]> features = new EnumMap<>(Feature.Type.class);

    public BowImage(SuperPixelMethond spMethod, StripMethod stripMethod, BufferedImage image, BufferedImage mask) {
        this.image = image;
        this.image4 = ImageUtils.resize(image, 4, false);
        this.sp4 = spMethod.generate(image4);

        if (mask != null) {
            this.mask = mask;
            this.mask4 = ImageUtils.resize(mask, 4, true);
            this.strip4 = stripMethod.strip(sp4, mask4);
        }
    }
}
