package com.github.levyfan.reid.util;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
public class ImageUtils {

    public static BufferedImage resize(BufferedImage image, int scale, boolean nearest) {
        BufferedImage target = new BufferedImage(
                image.getWidth()*scale, image.getHeight()*scale, image.getType());
        Graphics2D graphics2D = target.createGraphics();

        if (nearest) {
            graphics2D.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
            graphics2D.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_OFF);
        } else {
            graphics2D.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            graphics2D.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON);
        }
        graphics2D.setRenderingHint(
                RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY
        );
        graphics2D.setRenderingHint(
                RenderingHints.KEY_COLOR_RENDERING,
                RenderingHints.VALUE_COLOR_RENDER_QUALITY
        );
        graphics2D.setRenderingHint(
                RenderingHints.KEY_DITHERING,
                RenderingHints.VALUE_DITHER_ENABLE
        );

        graphics2D.drawImage(image, 0, 0, image.getWidth()*scale, image.getHeight()*scale, null);
        graphics2D.dispose();

        return target;
    }
}
