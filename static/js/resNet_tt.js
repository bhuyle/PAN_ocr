$(document).ready(function () {
    // Init
    $('.image-section-ResNet50').hide();
    $('.loaderResNet50').hide();
    $('#resultResNet_tt').hide();

    // Upload Preview
    function readURLResNet50(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreviewResNet50').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreviewResNet50').hide();
                $('#imagePreviewResNet50').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUploadResNet_pre").change(function () {
        $('.image-section-ResNet50').show();
        $('#btn-predict-ResNet_pre').show();
        $('#resultResNet_tt').text('');
        $('#resultResNet_tt').hide();
        $('#text18').text('')

        readURLResNet50(this);
    });

    // Predict
    $('#btn-predict-ResNet_pre').click(function () {
        var form_data = new FormData($('#upload-file-ResNet_pre')[0]);

        // Show loading animation
        $(this).hide();
        $('.loaderResNet50').show();
        // Make prediction by calling api /predictResNet50
        $.ajax({
            type: 'POST',
            url: './predictResNet_tt',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loaderResNet50').hide();
                $('#resultResNet_tt').fadeIn(600);
                $("#resultResNet_tt").attr("src","data:image/png;base64,"+ data.image);
                $('#text18').html("Number of words "+data.box+'<br>Time '+data.time)

            },
        });
    });

});